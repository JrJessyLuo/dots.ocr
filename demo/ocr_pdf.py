#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt


# ---------------------- PDF → PIL helpers ----------------------

def load_images_from_pdf(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Render PDF pages to PIL images.
    Prefers PyMuPDF (fitz); falls back to pdf2image if available.
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


# ---------------------- Model loading + inference ----------------------

def _ensure_local_rank():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

def _pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

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

def run_inference_on_pil(pil_img: Image.Image, prompt: str, model, processor, max_new_tokens: int = 4096) -> str:
    """
    Single (PIL image, prompt) → generated text.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": prompt},
        ],
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
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


# ---------------------- PDF Parser (multi-page) ----------------------

def parse_pdf(
    pdf_path: str,
    model,
    processor,
    prompt_mode: str,
    prompt_text: str,
    out_dir: Path,
    dpi: int = 200,
    max_new_tokens: int = 4096,
    num_threads: int = 1,
) -> List[Dict[str, Any]]:
    """
    Render a PDF to images, then run model on each page with given prompt.
    Returns list of records (one per page).
    """
    print(f"[info] loading pdf: {pdf_path}")
    images = load_images_from_pdf(pdf_path, dpi=dpi)
    total_pages = len(images)
    print(f"[info] PDF pages: {total_pages} (threads={num_threads})")

    pdf_stem = Path(pdf_path).stem
    records: List[Dict[str, Any]] = []

    def _work(page_idx_img):
        idx, img = page_idx_img
        t0 = time.time()
        try:
            text = run_inference_on_pil(
                pil_img=img,
                prompt=prompt_text,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
            )
            status = "ok"
        except Exception as e:
            text = f"[ERROR] {e}"
            status = "error"
        dt = time.time() - t0
        return {
            "file_path": pdf_path,
            "pdf_stem": pdf_stem,
            "page_no": idx,
            "prompt_mode": prompt_mode,
            "prompt": prompt_text,
            "output_text": text,
            "time_sec": dt,
            "status": status,
        }

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=min(num_threads, total_pages)) as ex:
            futs = [ex.submit(_work, (i, im)) for i, im in enumerate(images)]
            for fut in as_completed(futs):
                records.append(fut.result())
    else:
        for i, im in enumerate(images):
            records.append(_work((i, im)))

    # sort by page order
    records.sort(key=lambda r: r["page_no"])

    # write per-PDF JSONL (one line per page)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{pdf_stem}.{prompt_mode}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for r in records:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[info] saved: {out_jsonl}")
    return records


# ---------------------- CLI ----------------------

def main():
    _ensure_local_rank()

    ap = argparse.ArgumentParser(description="DotsOCR/Qwen-VL over PDFs (page-wise)")
    ap.add_argument("--model_path", default="./weights/DotsOCR", help="Path to model")
    ap.add_argument("--pdfs", required=True, help="PDF file, directory, or glob (e.g., '/path/*.pdf')")
    ap.add_argument("--out_dir", default="dotsocr_pdf_out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1, help="Page-level threads per prompt")
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
    print(f"[info] Found {len(pdfs)} PDF(s). Output: {Path(args.out_dir).resolve()}")

    # Load model once
    model, processor = load_model_and_processor(args.model_path)
    pdfs = pdfs[:1]

    # Run each PDF for each prompt_mode
    for pdf_path in pdfs:
        for prompt_mode, prompt in dict_promptmode_to_prompt.items():
            print(f"\n[run] PDF={pdf_path}  prompt_mode={prompt_mode}")
            parse_pdf(
                pdf_path=pdf_path,
                model=model,
                processor=processor,
                prompt_mode=prompt_mode,
                prompt_text=prompt,
                out_dir=Path(args.out_dir) / Path(pdf_path).stem,
                dpi=args.dpi,
                max_new_tokens=args.max_new_tokens,
                num_threads=args.threads,
            )

if __name__ == "__main__":
    main()
