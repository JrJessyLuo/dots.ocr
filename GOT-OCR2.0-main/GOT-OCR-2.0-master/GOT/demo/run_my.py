#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOT-Qwen → MinerU-like elements extractor (tables/figures/captions) from PDFs,
with per-PDF time costs.

Output layout (per PDF):
  <out_root>/<pdf_stem>/
    manifest.json
    mineru_elements.json

Additionally, a per-run timing file is written:
  <out_root>/got_time_cost.json

Usage:
  python got_elements_from_pdfs.py \
      --model-name /path/to/GOT/checkpoint \
      --pdfs "/path/to/dir/*.pdf" \
      --out-dir got_mineru_output \
      --dpi 200 --threads 2
"""

import os
import re
import json
import time
import uuid
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from html import escape

# --- GOT imports (your environment must have these) ---
from transformers import AutoTokenizer
from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import conv_templates, SeparatorStyle


# ========================= Small helpers =========================

def _safe_stem(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text.strip())
    return s[:150] or str(uuid.uuid4())[:8]


def load_images_from_pdf(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Render PDF pages to PIL images (prefer PyMuPDF; fallback pdf2image)."""
    try:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
        return pages
    except Exception as e:
        print(f"[warn] PyMuPDF failed: {e}; trying pdf2image ...")
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF {pdf_path}: {e}")


def omni_image_path(pdf_stem: str, page_idx: int) -> str:
    """OmniDocBench-style naming for page images (if you later save them)."""
    return f"{pdf_stem}.pdf_{page_idx}.jpg"


# ========================= Text → elements =========================

def _strip_tex(s: str) -> str:
    s = re.sub(r"\\\(|\\\)", "", s)
    s = re.sub(r"\^\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", s)  # \hline, \mathrm{..}, etc.
    s = s.replace("{", "").replace("}", "")
    return " ".join(s.split())


def _tabular_to_html(tabular_body: str) -> str:
    body = re.sub(r"(?mi)^\s*\\hline\s*$", "", tabular_body)
    body = re.sub(r"(?mi)^\s*\\cline\{[^}]+\}\s*$", "", body)
    rows = [r for r in re.split(r"\\\\\s*", body.strip()) if r.strip()]
    html_rows = []
    for r in rows:
        r = r.replace("\\hline", "")
        cells = [escape(_strip_tex(c.strip())) for c in r.split("&")]
        if any(cells):
            html_rows.append(cells)
    if not html_rows:
        return "<table><tbody></tbody></table>"

    def looks_header(cells: List[str]) -> bool:
        if not cells:
            return False
        alpha = sum(bool(re.search(r"[A-Za-z]", c)) for c in cells)
        num   = sum(bool(re.fullmatch(r"[-+]?(\d+(\.\d+)?)", c)) for c in cells)
        return alpha >= num

    head_html, body_html = "", ""
    if looks_header(html_rows[0]):
        head, body = html_rows[0], html_rows[1:]
        head_html = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead>"
        body_html = "<tbody>" + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body
        ) + "</tbody>"
    else:
        body_html = "<tbody>" + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in html_rows
        ) + "</tbody>"
    return f"<table>{head_html}{body_html}</table>"


def _markdown_table_to_html(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return ""
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    # remove separator line(s)
    data_lines = [ln for ln in lines[1:] if not re.fullmatch(
        r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", ln)]
    rows = [[escape(c.strip()) for c in ln.strip("|").split("|")] for ln in data_lines]
    html = "<table>"
    html += "<thead><tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr></thead>"
    html += "<tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody>"
    html += "</table>"
    return html


def build_elements_from_text(text: str, page_idx: int = 0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a single page's OCR text (possibly mixing plain text + LaTeX table blobs)
    into MinerU-like `elements`.
    """
    elements: List[Dict[str, Any]] = []

    # TABLE captions (e.g., "TABLE 2 - ...")
    caption_matches = list(re.finditer(r"(?mi)\bTABLE\s+\d+[^\\\n]*", text))
    captions = [{"span": m.span(), "text": _strip_tex(m.group(0)).strip()} for m in caption_matches]

    # LaTeX tabular blocks
    tab_blocks = list(re.finditer(r"\\begin\{tabular\}\{[^}]*\}(.+?)\\end\{tabular\}", text, re.S))
    for m in tab_blocks:
        body = m.group(1)
        html = _tabular_to_html(body)
        start = m.span()[0]
        cap_text = ""
        window = 400
        prev = [c for c in captions if c["span"][1] <= start and (start - c["span"][1]) <= window]
        nextc = [c for c in captions if c["span"][0] >= start and (c["span"][0] - start) <= window]
        if prev:
            cap_text = prev[-1]["text"]
        elif nextc:
            cap_text = nextc[0]["text"]

        elements.append({
            "page_idx": page_idx, "category_type": "table",
            "bbox_norm": None, "text": "", "html": html
        })
        if cap_text:
            elements.append({
                "page_idx": page_idx, "category_type": "table_caption",
                "bbox_norm": None, "text": cap_text, "html": ""
            })

    # Markdown pipe tables
    for blk in re.finditer(r"((?:^\s*\|.*\n){2,})", text, re.M):
        html = _markdown_table_to_html(blk.group(1))
        if html:
            elements.append({
                "page_idx": page_idx, "category_type": "table",
                "bbox_norm": None, "text": "", "html": html
            })

    # Figure captions
    for m in re.finditer(r"(?mi)\b(fig\.?|figure)\s*\d+[\.:)\-]\s*[^.\n]*.*?(?=(?:\n|$))", text):
        cap = _strip_tex(m.group(0)).strip()
        elements.append({
            "page_idx": page_idx, "category_type": "figure",
            "bbox_norm": None, "text": "", "html": ""
        })
        elements.append({
            "page_idx": page_idx, "category_type": "figure_caption",
            "bbox_norm": None, "text": cap, "html": ""
        })

    return {"elements": elements}


# ========================= GOT model inference =========================

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"

def got_prompt(image_size_tokens: int = 256, with_format: bool = False) -> str:
    """Build the same prompt pattern your GOT demo uses."""
    header = "OCR with format: " if with_format else "OCR: "
    return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_size_tokens + DEFAULT_IM_END_TOKEN + "\n" + header


def load_got_model(model_name: str):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=151643,
    ).eval()
    model.to(device="cuda", dtype=torch.bfloat16)
    # image processors
    proc_std = BlipImageEvalProcessor(image_size=1024)
    proc_hi  = BlipImageEvalProcessor(image_size=1024)
    return model, tokenizer, proc_std, proc_hi


def run_got_on_image(
    image: Image.Image,
    model,
    tokenizer,
    proc_std: BlipImageEvalProcessor,
    proc_hi: BlipImageEvalProcessor,
    with_format: bool = True,
    max_new_tokens: int = 4096,
) -> str:
    image_token_len = 256
    qs = got_prompt(image_token_len, with_format=with_format)

    conv = conv_templates["mpt"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # build image tensors
    image_tensor   = proc_std(image)       # (3, H, W)
    image_tensor_h = proc_hi(image.copy()) # (3, H, W)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(),
                     image_tensor_h.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
        )

    out = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if out.endswith(stop_str):
        out = out[:-len(stop_str)]
    return out.strip()

def filter_empty_elements(all_elements):
    # 1) record which pages have figure captions (so we can keep a placeholder figure if needed)
    caps_by_page = set()
    for el in all_elements:
        if el.get("category_type") == "figure_caption":
            caps_by_page.add(el.get("page_idx"))

    def keep(el):
        c = el.get("category_type")
        # Always keep non-figure types
        if c not in ("figure", "table"):
            return True

        # For tables: keep if we at least have HTML (or a bbox/text if you ever provide them)
        if c == "table":
            return bool(el.get("html")) or el.get("bbox_norm") is not None or bool(el.get("text"))

        # For figures: keep if there is *something* useful…
        has_payload = (el.get("bbox_norm") is not None) or bool(el.get("html")) or bool(el.get("text"))
        if has_payload:
            return True

        # …otherwise only keep a placeholder figure if a caption exists on the same page
        page_has_cap = el.get("page_idx") in caps_by_page
        return page_has_cap

    return [el for el in all_elements if keep(el)]


def filter_empty_elements(all_elements):
    # 1) record which pages have figure captions (so we can keep a placeholder figure if needed)
    caps_by_page = set()
    for el in all_elements:
        if el.get("category_type") == "figure_caption":
            caps_by_page.add(el.get("page_idx"))

    def keep(el):
        c = el.get("category_type")
        # Always keep non-figure types
        if c not in ("figure", "table"):
            return True

        # For tables: keep if we at least have HTML (or a bbox/text if you ever provide them)
        if c == "table":
            return bool(el.get("html")) or el.get("bbox_norm") is not None or bool(el.get("text"))

        # For figures: keep if there is *something* useful…
        has_payload = (el.get("bbox_norm") is not None) or bool(el.get("html")) or bool(el.get("text"))
        if has_payload:
            return True

        # …otherwise only keep a placeholder figure if a caption exists on the same page
        page_has_cap = el.get("page_idx") in caps_by_page
        return page_has_cap

    return [el for el in all_elements if keep(el)]



# ========================= Per-PDF pipeline =========================

def process_pdf(
    pdf_path: Path,
    out_root: Path,
    model,
    tokenizer,
    proc_std,
    proc_hi,
    dpi: int,
    threads: int,
) -> Tuple[Path, int, float]:
    """
    Process a single PDF into mineru_elements.json and manifest.json.
    Returns (mineru_path, total_pages, elapsed_sec).
    """
    t_pdf0 = time.time()

    pdf_stem = pdf_path.stem
    pdf_out = out_root / pdf_stem
    pdf_out.mkdir(parents=True, exist_ok=True)

    pages = load_images_from_pdf(str(pdf_path), dpi=dpi)
    total_pages = len(pages)
    print(f"[info] {pdf_path.name}: {total_pages} page(s)")

    def _worker(idx_img):
        idx, img = idx_img
        t0 = time.time()
        try:
            text = run_got_on_image(img, model, tokenizer, proc_std, proc_hi, with_format=True)
            parsed = build_elements_from_text(text, page_idx=idx)["elements"]
            # attach pdf/image_path so it is OmniDocBench-compatible
            for el in parsed:
                el.setdefault("pdf", pdf_stem)
                el.setdefault("image_path", omni_image_path(pdf_stem, idx))
            status = "ok"
        except Exception as e:
            parsed = []
            status = f"error: {e}"
        dt = time.time() - t0
        return idx, parsed, dt, status

    tasks = list(enumerate(pages))
    page_results: List[Tuple[int, List[Dict[str, Any]], float, str]] = []

    if threads > 1:
        from math import inf
        with ThreadPoolExecutor(max_workers=min(threads, max(1, total_pages))) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for f in as_completed(futs):
                page_results.append(f.result())
    else:
        for t in tasks:
            page_results.append(_worker(t))

    page_results.sort(key=lambda x: x[0])

    # flatten elements + collect per-page time
    all_elements: List[Dict[str, Any]] = []
    per_page_time = []
    per_page_status = []
    for idx, els, dt, st in page_results:
        all_elements.extend(els)
        per_page_time.append({"page_idx": idx, "time_sec": dt})
        per_page_status.append({"page_idx": idx, "status": st})

    all_elements = filter_empty_elements(all_elements)

    # save mineru_elements.json
    mineru_path = pdf_out / "mineru_elements.json"
    mineru_path.write_text(json.dumps(all_elements, ensure_ascii=False, indent=2), encoding="utf-8")

    # minimal manifest (add per-page timing/status for convenience)
    manifest = {
        "pdf": pdf_stem,
        "tables": [],     # elements already carry table HTML if any
        "images": [],
        "metrics": {
            "per_page_time_sec": per_page_time,
            "per_page_status": per_page_status,
        },
    }
    manifest_path = pdf_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_pdf0
    print(f"[ok]  {pdf_path.name} → {mineru_path}  (time: {elapsed:.2f}s)")
    return mineru_path, total_pages, elapsed


# ========================= CLI =========================

def collect_pdfs(spec: str) -> List[Path]:
    p = Path(spec)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.pdf"))
    # glob pattern
    return sorted(Path().glob(spec))


def main():
    ap = argparse.ArgumentParser(description="GOT-Qwen → MinerU-like elements for PDFs (with time costs)")
    ap.add_argument("--model-name", required=True, help="Path or hub id of GOT-Qwen model")
    ap.add_argument("--pdfs", required=True, help="PDF file, dir, or glob (e.g., '/data/*.pdf')")
    ap.add_argument("--out-dir", default="got_mineru_output", help="Output root")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI")
    ap.add_argument("--threads", type=int, default=1, help="Threads per PDF")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.pdfs)
    pdfs = pdfs[:10]
    if not pdfs:
        raise SystemExit(f"No PDFs found for: {args.pdfs}")
    print(f"[info] Found {len(pdfs)} PDF(s). Output root: {out_root.resolve()}")

    # load GOT once
    model, tokenizer, proc_std, proc_hi = load_got_model(args.model_name)

    successes, failures = [], []
    time_costs: Dict[str, float] = {}

    for pdf in pdfs:
        try:
            _, _, elapsed = process_pdf(
                pdf, out_root, model, tokenizer, proc_std, proc_hi,
                dpi=args.dpi, threads=args.threads
            )
            time_costs[pdf.name] = elapsed
            successes.append(pdf.name)
        except Exception as e:
            print(f"[err] {pdf.name}: {e}")
            time_costs[pdf.name] = 0.0
            failures.append(pdf.name)

    # Save per-PDF time costs
    time_cost_path = out_root / "got_time_cost.json"
    time_cost_path.write_text(
        json.dumps(time_costs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    

    print(f"\n[got-mineru] Saved per-PDF time costs to {time_cost_path}")

    # Summary
    print("\nSummary")
    print(f"  Success: {len(successes)}")
    print(f"  Failed : {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
