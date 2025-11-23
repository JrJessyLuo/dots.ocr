#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# If you use this from DotsOCR repo, keep this import; otherwise replace with your own prompts dict
from dots_ocr.utils import dict_promptmode_to_prompt  # optional; not strictly used here


# --------------------------- Prompt & naming ---------------------------

# def build_layout_prompt(page_idx: int) -> str:
#     """
#     JSON-only instruction for layout extraction (tables/figures + captions).
#     """
#     return f"""
# You are a document layout extractor for research PDFs.

# For THIS page image, find ALL:
#   - tables
#   - table captions
#   - figures
#   - figure captions

# For EACH region you detect, output an object with:
#   - "page_idx": integer page index (0-based). For this page, ALWAYS use {page_idx}.
#   - "category_type": one of ["table", "table_caption", "figure", "figure_caption"].
#   - "bbox_norm": [x0, y0, x1, y1] in NORMALIZED coordinates (top-left, bottom-right), each in [0, 1].
#         If you truly cannot estimate, use null.
#   - "text": plain text content.
#         * For captions: the full caption text (e.g., "Figure 1: ...").
#         * For tables: leave "" or a very short summary if needed.
#         * For figures: usually "".
#   - "html": for tables ONLY, reconstruct the table BODY as HTML:
#         <table> with <thead>/<tbody>/<tr>/<th>/<td>.
#         Do NOT include caption text in this HTML.
#         For non-table elements, set "html": "".

# Return ONLY a JSON object with the following structure:

# {{
#   "elements": [
#      {{
#        "page_idx": {page_idx},
#        "category_type": "table",
#        "bbox_norm": [0.1, 0.2, 0.9, 0.5],
#        "text": "",
#        "html": "<table>...</table>"
#      }},
#      {{
#        "page_idx": {page_idx},
#        "category_type": "table_caption",
#        "bbox_norm": [0.1, 0.15, 0.9, 0.2],
#        "text": "Table 1: Example caption.",
#        "html": ""
#      }},
#      {{
#        "page_idx": {page_idx},
#        "category_type": "figure",
#        "bbox_norm": [0.1, 0.55, 0.9, 0.9],
#        "text": "",
#        "html": ""
#      }},
#      {{
#        "page_idx": {page_idx},
#        "category_type": "figure_caption",
#        "bbox_norm": [0.1, 0.9, 0.9, 0.95],
#        "text": "Figure 1: Example caption.",
#        "html": ""
#      }}
#   ]
# }}

# If there are NO such elements, return {{"elements": []}}.

# IMPORTANT:
#   - Do NOT include any extra keys or commentary.
#   - The top-level object MUST have exactly one key "elements" with a list of objects.
# """.strip()


def omni_image_path(pdf_stem: str, page_idx: int) -> str:
    """Match OmniDocBench naming: <pdf_stem>.pdf_<page_idx>.jpg"""
    return f"{pdf_stem}.pdf_{page_idx}.jpg"


# --------------------------- Utilities ---------------------------

def _ensure_local_rank():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"


def _pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def safe_json_from_text(s: str) -> Dict[str, Any]:
    """
    Be tolerant to models returning code fences or stray text.
    """
    if not s:
        return {"elements": []}
    # Strip code fences
    s = s.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "elements" in obj and isinstance(obj["elements"], list):
            return obj
    except Exception:
        pass
    # Fallback: extract the largest {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "elements" in obj and isinstance(obj["elements"], list):
                return obj
        except Exception:
            return {"elements": []}
    return {"elements": []}


# --------------------------- PDF → images ---------------------------

def load_images_from_pdf(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Render PDF pages to PIL images. Prefers PyMuPDF; falls back to pdf2image.
    """
    images = []
    try:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        return images
    except Exception as e:
        print(f"[warn] PyMuPDF render failed ({e}); trying pdf2image ...")

    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF {pdf_path}: {e}")


# --------------------------- Model load + inference ---------------------------

def load_model_and_processor(model_path: str):
    dtype = _pick_dtype()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[warn] flash_attention_2 not available ({e}); using default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


import re
import json
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch, time

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```$', '', s)
    return s.strip()

def _parse_output_text_to_list(text: str) -> List[Dict[str, Any]]:
    """
    Accepts model text like:
      '[{"bbox":[...],"category":"Text","text":".."}, ...]'
    Possibly wrapped in code fences or with stray text.
    Returns a list of dicts; empty list on failure.
    """
    if not text:
        return []
    s = _strip_code_fences(text)

    # Try direct list parse
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        # Sometimes it's a dict with a list under a key
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
    except Exception:
        pass

    # Fallback: extract the largest [...] block
    m = re.search(r'\[.*\]', s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []

def _norm_bbox(bbox, w: int, h: int):
    """
    bbox: [x0,y0,x1,y1] in pixels → [nx0,ny0,nx1,ny1] in [0,1]
    Returns None if invalid.
    """
    if (not isinstance(bbox, (list, tuple))) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, bbox)
        if w <= 0 or h <= 0:
            return None
        return [x0 / w, y0 / h, x1 / w, y1 / h]
    except Exception:
        return None

def _category_map(cat: str, text: str) -> str:
    """
    Map raw detector categories to your target types.
    """
    c = (cat or "").strip().lower()
    if c == "picture":
        return "figure"
    if c in {"text", "section-header"}:
        t = (text or "").strip().lower()
        if t.startswith("fig") or t.startswith("figure"):
            return "figure_caption"
        # add more heuristics if needed:
        # if t.startswith("table") or t.startswith("tab"):
        #     return "table_caption"
        return ""  # ignore other free text
    # If your upstream can emit table categories, add them here:
    # if c in {"table", "table body"}: return "table"
    return ""  # ignore unrecognized categories

def run_layout_json_on_image(
    pil_img: Image.Image,
    page_idx: int,
    model,
    processor,
    max_new_tokens: int = 4096,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run DotsOCR/Qwen-VL on (image, layout_prompt) and convert the returned
    JSON-like list of {bbox, category, text} into your target 'elements' schema.
    """
    # Use your existing prompt (JSON-only not required if upstream returns that list)
    prompt = dict_promptmode_to_prompt['prompt_layout_all_en']

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": prompt},
        ],
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[chat_text],
        images=[pil_img],
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    elapsed = time.time() - t0

    raw_text = texts[0] if texts else ""
    items = _parse_output_text_to_list(raw_text)

    W, H = pil_img.size
    elements: List[Dict[str, Any]] = []

    for rec in items:
        bbox = rec.get("bbox")
        cat  = rec.get("category", "")
        txt  = rec.get("text", "")

        mapped = _category_map(cat, txt)
        if not mapped:
            continue

        bbox_norm = _norm_bbox(bbox, W, H)
        if mapped == "figure":
            elements.append({
                "page_idx": page_idx,
                "category_type": "figure",
                "bbox_norm": bbox_norm,   # may be None if invalid
                "text": "",
                "html": ""
            })
        elif mapped == "figure_caption":
            elements.append({
                "page_idx": page_idx,
                "category_type": "figure_caption",
                "bbox_norm": bbox_norm,
                "text": txt or "",
                "html": ""
            })
        # If you add table/table_caption mapping above, handle them here similarly.

    meta = {
        "input_tokens": 0.0,     # local models usually don't provide token usage
        "output_tokens": 0.0,
        "total_tokens": 0.0,
        "time_sec": float(elapsed),
        "raw_text_len": float(len(raw_text)),
    }
    return elements, meta



# --------------------------- PDF parser (multi-page) ---------------------------

def parse_pdf_to_elements(
    pdf_path: str,
    model,
    processor,
    out_dir: Path,
    dpi: int = 200,
    max_new_tokens: int = 4096,
    num_threads: int = 1,
) -> Path:
    """
    Render PDF → images, extract layout JSON per page, save page images with
    Omni naming, and write a single <pdf_stem>.elements.jsonl (one line/page).
    """
    print(f"[info] loading pdf: {pdf_path}")
    images = load_images_from_pdf(pdf_path, dpi=dpi)
    total_pages = len(images)
    print(f"[info] PDF pages: {total_pages} (threads={num_threads})")

    pdf_stem = Path(pdf_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def _worker(idx_img: Tuple[int, Image.Image]) -> Dict[str, Any]:
        idx, img = idx_img

        # Save page image with Omni naming
        img_name = omni_image_path(pdf_stem, idx)
        img_path = img_dir / img_name
        try:
            img.save(img_path, format="JPEG", quality=90)
        except Exception:
            # fallback: ensure RGB
            img = img.convert("RGB")
            img.save(img_path, format="JPEG", quality=90)

        try:
            elements, meta = run_layout_json_on_image(
                pil_img=img,
                page_idx=idx,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"[warn] inference failed on page {idx}: {e}")
            elements, meta = [], {"time_sec": 0.0}

        # attach pdf + image_path like your GPT-4o pipeline
        for el in elements:
            el.setdefault("pdf", pdf_stem)
            el.setdefault("image_path", f"{pdf_stem}.pdf_{idx}.jpg")

        rec = {
            "pdf": pdf_stem,
            "pdf_path": str(pdf_path),
            "page_idx": idx,
            "image_path": str(img_path.name),
            "elements": elements,
            "metrics": meta,
        }
        return rec

    results: List[Dict[str, Any]] = []
    tasks = list(enumerate(images))
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=min(num_threads, total_pages)) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for t in tasks:
            results.append(_worker(t))

    results.sort(key=lambda r: r["page_idx"])

    all_elements: List[Dict[str, Any]] = []
    for r in results:
        els = r.get("elements", [])
        if isinstance(els, list):
            all_elements.extend(els)

    # Write JSONL: one record per page
    out_json = out_dir / f"mineru_elements.json"
    out_json.write_text(
        json.dumps({"elements": all_elements}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[info] saved elements → {out_json}")
    return out_json


# --------------------------- CLI ---------------------------

def main():
    _ensure_local_rank()

    ap = argparse.ArgumentParser(description="DotsOCR/Qwen-VL layout extraction over PDFs (page-wise)")
    ap.add_argument("--model_path", default="./weights/DotsOCR", help="Path to DotsOCR/Qwen-VL model")
    ap.add_argument("--pdfs", required=True, help="PDF file, directory, or glob (e.g., '/path/*.pdf')")
    ap.add_argument("--out_dir", default="dotsocr_layout_out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1, help="Page-level threads")
    args = ap.parse_args()

    # Collect PDFs
    p = Path(args.pdfs)
    if p.is_file():
        pdfs = [str(p)]
    elif p.is_dir():
        pdfs = [str(x) for x in sorted(p.glob("*.pdf"))]
    else:
        pdfs = [str(x) for x in sorted(Path().glob(args.pdfs))]
    if not pdfs:
        raise SystemExit(f"No PDFs found for: {args.pdfs}")

    print(f"[info] Found {len(pdfs)} PDF(s). Output root: {Path(args.out_dir).resolve()}")

    # Load the model once
    model, processor = load_model_and_processor(args.model_path)
    pdfs = pdfs[:1]

    for pdf_path in pdfs:
        pdf_out_dir = Path(args.out_dir) / Path(pdf_path).stem
        parse_pdf_to_elements(
            pdf_path=pdf_path,
            model=model,
            processor=processor,
            out_dir=pdf_out_dir,
            dpi=args.dpi,
            max_new_tokens=args.max_new_tokens,
            num_threads=args.threads,
        )


if __name__ == "__main__":
    main()
