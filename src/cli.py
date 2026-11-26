import argparse
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any, Optional

import ollama
from PIL import Image
from pydantic import BaseModel, BeforeValidator, Field


# ==========================================
# 1. UNIFIED SCHEMA & TYPE COERCION
# ==========================================
def _to_opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return ", ".join(parts) or None
    return str(v).strip() or None


def _to_str_list(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _to_opt_float(v: Any) -> Optional[float]:
    if v is None or v == "" or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(v))
    return float(m.group(0).replace(",", "")) if m else None


def _to_obj_list(v: Any) -> list[Any]:
    if not v:
        return []
    if isinstance(v, dict):
        return [v]
    return [x for x in v if isinstance(x, (dict, BaseModel))] if isinstance(v, list) else []


def _to_details(v: Any) -> list[Any]:
    if not v:
        return []
    if isinstance(v, dict):
        return [
            {"attribute": str(k), "value": _to_opt_str(val) or ""}
            for k, val in v.items()
            if not str(k).startswith("$")
        ]
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, CategoryDetail) or (isinstance(item, dict) and "attribute" in item and "value" in item):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(
                    {"attribute": str(k), "value": _to_opt_str(val) or ""}
                    for k, val in item.items()
                    if not str(k).startswith("$")
                )
        return out
    return []


OptStr = Annotated[Optional[str], BeforeValidator(_to_opt_str)]
ReqStr = Annotated[str, BeforeValidator(lambda v: _to_opt_str(v) or "")]
StrList = Annotated[list[str], BeforeValidator(_to_str_list)]
OptFloat = Annotated[Optional[float], BeforeValidator(_to_opt_float)]


class PricingOrOffer(BaseModel):
    label: Annotated[str, BeforeValidator(lambda v: _to_opt_str(v) or "Offer")] = "Offer"
    details: ReqStr = ""
    amount: OptFloat = None
    conditions: OptStr = None


class ShopOrDealer(BaseModel):
    name: ReqStr = ""
    locations: StrList = Field(default_factory=list)
    city: OptStr = None
    phones: StrList = Field(default_factory=list)


class CategoryDetail(BaseModel):
    attribute: ReqStr
    value: ReqStr = ""


class ContactInfo(BaseModel):
    phones: StrList = Field(default_factory=list)
    website: OptStr = None
    email: OptStr = None
    other_cta: OptStr = None


class UnifiedAd(BaseModel):
    category: Annotated[str, BeforeValidator(lambda v: _to_opt_str(v) or "Other")] = "Other"
    brand_or_company: OptStr = None
    product_or_service: OptStr = None
    headline: OptStr = None
    pricing_and_offers: Annotated[list[PricingOrOffer], BeforeValidator(_to_obj_list)] = Field(default_factory=list)
    key_features: StrList = Field(default_factory=list)
    category_specific_details: Annotated[list[CategoryDetail], BeforeValidator(_to_details)] = Field(default_factory=list)
    shops_or_dealers: Annotated[list[ShopOrDealer], BeforeValidator(_to_obj_list)] = Field(default_factory=list)
    contacts: Annotated[
        ContactInfo,
        BeforeValidator(lambda v: v if isinstance(v, (dict, ContactInfo)) else {}),
    ] = Field(default_factory=ContactInfo)
    languages: StrList = Field(default_factory=list)
    uncertain_fields: StrList = Field(default_factory=list)


# ==========================================
# 2. PROMPT & FLAT JSON TEMPLATE
# ==========================================
SCHEMA_TEMPLATE = """{
  "category": "string (e.g., Automobiles, Real Estate, Education, Jewelry, Electronics, Healthcare, Retail, Jobs)",
  "brand_or_company": "string or null",
  "product_or_service": "string or null",
  "headline": "string or null",
  "pricing_and_offers": [
    {
      "label": "string (e.g., Price, Starting Price, EMI, Discount, Benefits, Financing)",
      "details": "string (exact offer/price text with ₹ symbol)",
      "amount": "number or null",
      "conditions": "string or null"
    }
  ],
  "key_features": ["string"],
  "category_specific_details": [
    {"attribute": "string (e.g., Engine, Fuel, RERA ID, Purity, Batch Date)", "value": "string"}
  ],
  "shops_or_dealers": [
    {"name": "string", "locations": ["string"], "city": "string or null", "phones": ["string"]}
  ],
  "contacts": {
    "phones": ["string"],
    "website": "string or null",
    "email": "string or null (comma-separated string if multiple)",
    "other_cta": "string or null"
  },
  "languages": ["ISO code string (e.g., en, hi, or, ta)"],
  "uncertain_fields": ["schema field name string"]
}"""

PROMPT = f"""Read this newspaper advertisement image directly (English or any Indian language) and return ONE JSON object following this exact structure:
{SCHEMA_TEMPLATE}

Rules:
1. Extract only what is visually present in the image; leave absent fields null or [].
2. Keep values concise—summarize specs and skip long legal disclaimers or T&C footers.
3. Read numbers, currency symbols (₹), phone numbers, URLs, and emails accurately directly from the image.
4. For regional-language ads, preserve brand names and phone numbers accurately, and write categories, headlines, and features in clear English.
5. Put unique industry-specific facts in `category_specific_details` without repeating `key_features`.
6. Separate every distinct shop/dealer with its matching locations, city, and phone numbers.
7. In `uncertain_fields`, list only schema field names where text was blurry or unreadable."""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ==========================================
# 3. IMAGE & VLM EXTRACTION
# ==========================================
def resize_image(image_path: Path, max_dim: int) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def extract_ad(client: ollama.Client, image_path: Path, model_name: str, max_dim: int = 1536) -> tuple[UnifiedAd, float, int]:
    attempts = [
        {"dim": max_dim, "temp": 0.0, "repeat_penalty": 1.1},
        {"dim": 1024, "temp": 0.2, "repeat_penalty": 1.15},
    ]

    last_err: Optional[Exception] = None
    for idx, cfg in enumerate(attempts, start=1):
        messages = [{"role": "user", "content": PROMPT, "images": [resize_image(image_path, cfg["dim"])]}]
        if "qwen3" in model_name.lower() and "instruct" not in model_name.lower():
            messages.append({"role": "assistant", "content": "<think>\n</think>\n"})

        start = time.perf_counter()
        stream = client.chat(
            model=model_name,
            messages=messages,
            format="json",
            stream=True,
            keep_alive="30m",
            options={
                "temperature": cfg["temp"],
                "repeat_penalty": cfg["repeat_penalty"],
                "num_ctx": 8192,
                "num_predict": 3072,
            },
        )

        chunks, done_reason = [], None
        for chunk in stream:
            if chunk.message.content:
                chunks.append(chunk.message.content)
            if chunk.done:
                done_reason = getattr(chunk, "done_reason", None)

        elapsed = round(time.perf_counter() - start, 2)
        raw = re.sub(r"<think>.*?</think>", "", "".join(chunks), flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()

        try:
            if not raw:
                raise RuntimeError(f"Empty response (done_reason={done_reason})")
            return UnifiedAd.model_validate_json(raw), elapsed, idx
        except Exception as e:
            last_err = RuntimeError(f"Attempt {idx} failed (done_reason={done_reason}): {e}")
            if idx < len(attempts):
                print(f"[retry {idx + 1}: {done_reason or 'parse error'}]", end=" ", flush=True)

    raise last_err if last_err else RuntimeError("Extraction failed")


def save_results(out_path: Path, model_name: str, source_dir: str, ads_map: dict) -> None:
    ok = [v for v in ads_map.values() if v.get("status") == "success"]
    times = [v["vlm_time_sec"] for v in ok if isinstance(v.get("vlm_time_sec"), (int, float))]
    payload = {
        "model": model_name,
        "source_dir": source_dir,
        "total_processed": len(ads_map),
        "success_count": len(ok),
        "error_count": len(ads_map) - len(ok),
        "avg_vlm_time_sec": round(sum(times) / len(times), 2) if times else 0.0,
        "ads": dict(sorted(ads_map.items())),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ==========================================
# 4. CLI ENTRYPOINT
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract newspaper ads with a chosen Ollama VLM.")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Ollama model name (e.g., qwen3-vl:8b, qwen2.5vl:7b)")
    parser.add_argument("--input", default=r"dataset\images", help="Folder of images or a single image path")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to outputs/extracted_ads_<model>.json)")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "[https://increasingly-retail-dual-tell.trycloudflare.com](https://increasingly-retail-dual-tell.trycloudflare.com)"),
        help="Kaggle Ollama tunnel URL",
    )
    parser.add_argument("--max-dim", type=int, default=1536, help="Max image dimension for attempt 1")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    parser.add_argument("--force", action="store_true", help="Ignore saved progress and re-run all images")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    images = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {input_path}")
        return

    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out_path = Path(args.out) if args.out else Path("outputs") / f"extracted_ads_{safe_model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ads_map: dict = {}
    if out_path.exists() and not args.force:
        try:
            ads_map = json.loads(out_path.read_text(encoding="utf-8")).get("ads", {})
            done = sum(1 for v in ads_map.values() if v.get("status") == "success")
            print(f"Resuming {out_path} ({done} already completed).")
        except Exception:
            print(f"Could not read {out_path}; starting fresh.")

    match = re.search(r"https?://[^\s\)\]]+", args.host)
    clean_host = match.group(0) if match else args.host.strip()
    client = ollama.Client(host=clean_host, timeout=180.0)

    print(f"Running {args.model} via {clean_host} on {len(images)} images -> {out_path}\n")
    batch_start = time.perf_counter()

    for idx, img_path in enumerate(images, start=1):
        image_id = img_path.stem
        if not args.force and ads_map.get(image_id, {}).get("status") == "success":
            print(f"[{idx}/{len(images)}] SKIP  {image_id}")
            continue

        print(f"[{idx}/{len(images)}] RUN   {image_id}...", end=" ", flush=True)
        try:
            ad, sec, tries = extract_ad(client, img_path, args.model, max_dim=args.max_dim)
            ads_map[image_id] = {
                "image_id": image_id,
                "filename": img_path.name,
                "status": "success",
                "attempt": tries,
                "vlm_time_sec": sec,
                "data": ad.model_dump(),
            }
            print(f"OK ({sec}s, try {tries}) | {ad.category} | {ad.brand_or_company or 'N/A'}")
        except Exception as e:
            ads_map[image_id] = {
                "image_id": image_id,
                "filename": img_path.name,
                "status": "error",
                "error": str(e),
            }
            print(f"ERROR: {e}")

        save_results(out_path, args.model, str(input_path), ads_map)

    total_sec = round(time.perf_counter() - batch_start, 1)
    ok = sum(1 for v in ads_map.values() if v.get("status") == "success")
    print(f"\nFinished in {total_sec}s | Success: {ok}/{len(images)} | Saved: {out_path}")


if __name__ == "__main__":
    main()