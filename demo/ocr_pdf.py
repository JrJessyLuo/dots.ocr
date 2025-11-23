#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DotsOCR/Qwen-VL → MinerU-like extractor (tables/figures/captions) using your layout prompt.

Output per PDF:
  <out_root>/<pdf_stem>/
    tables/
      p000_t0.html
      p000_t0.csv
      ...
    images/
      <saved page jpegs>
    manifest.json
    mineru_elements.json

Additionally, a per-run timing file is written:
  <out_root>/dotsocr_time_cost.json

Usage:
  python dotsocr_mineru_like_extract.py \
    --model_path /path/to/model \
    --pdfs "/path/to/dir/*.pdf" \
    --out_dir out_dir \
    --dpi 200 \
    --threads 2
"""

import os
import re
import json
import time
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor

# --------------------------- Prompt ---------------------------

PROMPT_LAYOUT = (
    "Please output the layout information from the PDF image, including each layout "
    "element's bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', "
    "'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)

# --------------------------- Omni naming ---------------------------

def omni_image_path(pdf_stem: str, page_idx: int) -> str:
    return f"{pdf_stem}.pdf_{page_idx}.jpg"

def sanitize_stem(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text.strip())
    return s[:150] or str(uuid.uuid4())[:8]

# --------------------------- Utilities ---------------------------

def _ensure_local_rank():
    os.environ.setdefault("LOCAL_RANK", "0")

def _pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```$', '', s)
    return s.strip()

def _looks_table_caption(text: str) -> bool:
    s = (text or "").strip().lower()
    return s.startswith("table ") or s.startswith("table:") or s.startswith("tab ") or s.startswith("tab.")

def _looks_figure_caption(text: str) -> bool:
    s = (text or "").strip().lower()
    return s.startswith("figure ") or s.startswith("figure:") or s.startswith("fig ") or s.startswith("fig.")

# --------------------------- PDF → PIL ---------------------------

def load_images_from_pdf(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    try:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        images = []
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

def run_model_on_image(
    pil_img: Image.Image,
    prompt: str,
    model,
    processor,
    max_new_tokens: int = 4096,
) -> str:
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

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return texts[0] if texts else ""

# --------------------------- Parse model output ---------------------------

def parse_detector_output(text: str) -> List[Dict[str, Any]]:
    """
    Accepts outputs like:
      '[{"bbox":[...],"category":"Text","text":".."}, ...]'
    OR the odd case: "['[{"bbox":...}]']"
    OR with code-fences.
    Returns: list of dicts (possibly empty).
    """
    s = _strip_code_fences(text)
    if not s:
        return []

    # Try direct JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "elements" in obj and isinstance(obj["elements"], list):
            return obj["elements"]
    except Exception:
        pass

    # Try to handle a quoted big list (e.g., "['[ {..} ]']")
    # Find the largest [...] block
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            # Sometimes quotes are single quotes; try a gentle normalization
            try:
                candidate2 = candidate.replace("'", '"')
                obj = json.loads(candidate2)
                if isinstance(obj, list):
                    return obj
            except Exception:
                return []

    return []

# --------------------------- Convert → MinerU-style elements ---------------------------

def to_mineru_elements(
    raw_items: List[Dict[str, Any]],
    page_idx: int,
    img_w: int,
    img_h: int,
    pdf_stem: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (eval_elements, table_artifacts), where:
      - eval_elements: flat list of mineru elements (table / table_caption / figure / figure_caption)
      - table_artifacts: list of dict {page_idx, html, caption} for saving to disk/manifest
    NOTE: To match the Mistral pipeline exactly, we force bbox_norm=None.
    """
    eval_elems: List[Dict[str, Any]] = []
    table_artifacts: List[Dict[str, Any]] = []

    for rec in raw_items:
        category = (rec.get("category") or "").strip()
        text = rec.get("text", "")

        # Strict parity with Mistral: DO NOT emit normalized boxes
        bbox_norm = None

        if category == "Picture":
            eval_elems.append({
                "pdf": pdf_stem,
                "image_path": omni_image_path(pdf_stem, page_idx),
                "page_idx": page_idx,
                "category_type": "figure",
                "bbox_norm": bbox_norm,
                "text": "",
                "html": "",
            })

        elif category == "Caption":
            if _looks_table_caption(text):
                eval_elems.append({
                    "pdf": pdf_stem,
                    "image_path": omni_image_path(pdf_stem, page_idx),
                    "page_idx": page_idx,
                    "category_type": "table_caption",
                    "bbox_norm": bbox_norm,
                    "text": text or "",
                    "html": "",
                })
            elif _looks_figure_caption(text):
                eval_elems.append({
                    "pdf": pdf_stem,
                    "image_path": omni_image_path(pdf_stem, page_idx),
                    "page_idx": page_idx,
                    "category_type": "figure_caption",
                    "bbox_norm": bbox_norm,
                    "text": text or "",
                    "html": "",
                })
            else:
                # ambiguous caption → default to figure_caption
                eval_elems.append({
                    "pdf": pdf_stem,
                    "image_path": omni_image_path(pdf_stem, page_idx),
                    "page_idx": page_idx,
                    "category_type": "figure_caption",
                    "bbox_norm": bbox_norm,
                    "text": text or "",
                    "html": "",
                })

        elif category == "Table":
            html = text or ""   # per prompt, tables must be HTML
            eval_elems.append({
                "pdf": pdf_stem,
                "image_path": omni_image_path(pdf_stem, page_idx),
                "page_idx": page_idx,
                "category_type": "table",
                "bbox_norm": bbox_norm,
                "text": "",
                "html": html,
            })
            table_artifacts.append({
                "page_idx": page_idx,
                "html": html,
                "caption": None,
            })

        # Other categories ignored for MinerU eval parity.

    return eval_elems, table_artifacts

# --------------------------- Per-PDF processing ---------------------------

def process_pdf(
    pdf_path: Path,
    out_root: Path,
    model,
    processor,
    dpi: int,
    threads: int,
    max_new_tokens: int,
) -> float:
    """
    Process a single PDF into mineru_elements.json and manifest.json.
    Returns the elapsed seconds for this PDF.
    """
    t_pdf0 = time.time()

    pdf_stem = pdf_path.stem
    pdf_out = out_root / pdf_stem
    img_dir = pdf_out / "images"
    tab_dir = pdf_out / "tables"
    img_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    pages = load_images_from_pdf(str(pdf_path), dpi=dpi)
    total_pages = len(pages)
    print(f"[info] {pdf_path.name}: {total_pages} page(s)")

    manifest = {
        "pdf": pdf_stem,
        "tables": [],
        "images": [],
    }
    all_eval_elements: List[Dict[str, Any]] = []

    def _worker(idx_img):
        idx, img = idx_img
        # save page jpeg (OmniDocBench naming)
        img_name = omni_image_path(pdf_stem, idx)
        img_path = img_dir / img_name
        try:
            img.save(img_path, format="JPEG", quality=90)
        except Exception:
            img = img.convert("RGB")
            img.save(img_path, format="JPEG", quality=90)

        t0 = time.time()
        try:
            raw = run_model_on_image(img, PROMPT_LAYOUT, model, processor, max_new_tokens=max_new_tokens)
            items = parse_detector_output(raw)
            eval_elems, table_artifacts = to_mineru_elements(items, idx, *img.size, pdf_stem)
            status = "ok"
        except Exception as e:
            eval_elems, table_artifacts = [], []
            status = f"error: {e}"
        dt = time.time() - t0

        # attach simple image bookkeeping (no raster extraction for figures here)
        for el in eval_elems:
            if el["category_type"] == "figure":
                manifest["images"].append({
                    "page_idx": idx,
                    "bbox": None,
                    "bbox_norm": None,  # parity: keep None
                    "path": None,
                    "caption": None,
                    "footnote": "",
                })

        # save tables (HTML + CSV)
        for i, t in enumerate(table_artifacts):
            html = t.get("html") or ""
            if not html.strip():
                continue
            stem = sanitize_stem(f"p{idx:03d}_t{i}")
            html_file = tab_dir / f"{stem}.html"
            html_file.write_text(html, encoding="utf-8")
            csv_path = None
            try:
                dfs = pd.read_html(str(html_file))
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    csv_file = tab_dir / f"{stem}.csv"
                    combined.to_csv(csv_file, index=False)
                    csv_path = str(csv_file)
            except Exception as e:
                print(f"[warn] pandas.read_html failed on {html_file.name}: {e}")

            manifest["tables"].append({
                "page_idx": idx,
                "bbox": None,
                "bbox_norm": None,  # parity: keep None
                "html": str(html_file),
                "csv": csv_path,
                "caption": "",
                "footnote": "",
            })

        return idx, eval_elems, dt, status

    tasks = list(enumerate(pages))
    page_results: List[Tuple[int, List[Dict[str, Any]], float, str]] = []

    if threads > 1:
        with ThreadPoolExecutor(max_workers=min(threads, total_pages)) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for f in as_completed(futs):
                page_results.append(f.result())
    else:
        for t in tasks:
            page_results.append(_worker(t))

    page_results.sort(key=lambda x: x[0])

    for _, elems, _, _ in page_results:
        all_eval_elements.extend(elems)

    # write outputs
    pdf_out.mkdir(parents=True, exist_ok=True)
    (pdf_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (pdf_out / "mineru_elements.json").write_text(json.dumps(all_eval_elements, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_pdf0
    print(f"[ok] {pdf_path.name}: {len(all_eval_elements)} elements → {pdf_out/'mineru_elements.json'}  (time: {elapsed:.2f}s)")
    return elapsed

# --------------------------- CLI ---------------------------

def collect_pdfs(spec: str) -> List[Path]:
    p = Path(spec)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.pdf"))
    return sorted(Path().glob(spec))

def main():
    _ensure_local_rank()

    ap = argparse.ArgumentParser(description="DotsOCR/Qwen-VL → MinerU-like elements from PDFs (Mistral-parity)")
    ap.add_argument("--model_path", required=True, help="Path or hub id to the vision-language model")
    ap.add_argument("--pdfs", required=True, help="PDF file, dir, or glob (e.g., '/data/*.pdf')")
    ap.add_argument("--out_dir", default="dotsocr_layout_out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1, help="Page-level threads per PDF")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.pdfs)[:10]
    if not pdfs:
        raise SystemExit(f"No PDFs found for: {args.pdfs}")

    print(f"[info] Found {len(pdfs)} PDF(s). Output root: {out_root.resolve()}")
    print("[info] Note: HF warns about slow image processor; that’s fine. "
          "Set `use_fast=False` on processor if you want to lock old behavior.")

    model, processor = load_model_and_processor(args.model_path)

    successes, failures = [], []
    time_costs: Dict[str, float] = {}

    for pdf in tqdm(pdfs, total=len(pdfs), desc="PDFs"):
        t0 = time.time()
        try:
            elapsed = process_pdf(pdf, out_root, model, processor, dpi=args.dpi, threads=args.threads, max_new_tokens=args.max_new_tokens)
            time_costs[pdf.name] = float(elapsed)
            successes.append(pdf.name)
        except Exception as e:
            print(f"[err] {pdf.name}: {e}")
            time_costs[pdf.name] = 0.0
            failures.append(pdf.name)

    # Save per-PDF time costs (parity with Mistral script)
    time_cost_path = out_root / "dotsocr_time_cost.json"
    time_cost_path.write_text(json.dumps(time_costs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[dotsocr-mineru] Saved per-PDF time costs to {time_cost_path}")

    print("\nSummary")
    print(f"  Success: {len(successes)}")
    print(f"  Failed : {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - {f}")

if __name__ == "__main__":
    main()
