import argparse
import base64
import json
import math
import os
from pathlib import Path

MODEL = "qwen3-vl:8b"


def report_timings():
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_files = sorted(outputs_dir.glob("extracted_ads_*.json"))
    if not output_files:
        raise FileNotFoundError(f"No extracted_ads_*.json files found in {outputs_dir}")

    totals = {}
    for output_file in output_files:
        with output_file.open(encoding="utf-8") as f:
            result = json.load(f)

        model = result.get("model", output_file.stem)
        ads = result.get("ads")
        if not isinstance(ads, dict):
            raise ValueError(f"Expected an 'ads' object in {output_file}")

        model_total = 0.0
        timed_ads = 0
        missing_times = 0
        for ad in ads.values():
            if not isinstance(ad, dict) or ad.get("status") != "success":
                continue
            elapsed = ad.get("vlm_time_sec")
            if (
                isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and math.isfinite(elapsed)
                and elapsed >= 0
            ):
                model_total += elapsed
                timed_ads += 1
            else:
                missing_times += 1

        model_entry = totals.setdefault(model, {"seconds": 0.0, "timed_ads": 0, "missing_times": 0})
        model_entry["seconds"] += model_total
        model_entry["timed_ads"] += timed_ads
        model_entry["missing_times"] += missing_times
        print(
            f"{model} ({output_file.name}): {model_total:.2f} seconds "
            f"across {timed_ads} successful ads"
        )
        if missing_times:
            print(f"  Warning: {missing_times} successful ads have no valid timing")

    total_seconds = sum(entry["seconds"] for entry in totals.values())
    print("\nTotals by model:")
    for model, entry in totals.items():
        print(f"  {model}: {entry['seconds']:.2f} seconds")
    print(f"Total time across all models: {total_seconds:.2f} seconds")


def encode_image(path):
    """Read an image file and return its base64 string (what Ollama expects)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError as e:
        print(f"  [!] cannot read image '{path}': {e}")
        return None


def stream_chat(base_url, history, headers):
    """Send history to /api/chat, stream the answer to the terminal, return full text."""
    import requests

    payload = {"model": MODEL, "messages": history, "stream": True}
    full = []
    try:
        with requests.post(f"{base_url}/api/chat", json=payload,
                           stream=True, timeout=600, headers=headers) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                print(chunk, end="", flush=True)
                full.append(chunk)
                if data.get("done"):
                    break
    except requests.RequestException as e:
        print(f"\n  [!] connection problem: {e}")
        print("      Is the Kaggle notebook still running? The tunnel URL changes")
        print("      every session - grab the fresh one and restart chatbot.py.")
        return None
    print()
    return "".join(full)


def run_chat(base_url_arg=None):
    # Endpoint: CLI arg > env var > localhost fallback
    base_url = "https://energy-essentials-well-sterling.trycloudflare.com"
    if base_url_arg:
        base_url = base_url_arg.rstrip("/")
    elif os.environ.get("OLLAMA_URL"):
        base_url = os.environ["OLLAMA_URL"].rstrip("/")

    # ngrok free tunnels require this header on non-browser requests
    headers = {"ngrok-skip-browser-warning": "true"} if "ngrok" in base_url else {}

    print(f"Chatting with {MODEL} at: {base_url}")
    print("Type /img file.jpg to attach an image, /reset to clear, /quit to exit.\n")

    history = []      # conversation so far
    pending_imgs = [] # base64 images waiting to be sent

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user in ("/quit", "/exit", "q"):
            print("Bye!")
            break
        if user == "/reset":
            history, pending_imgs = [], []
            print("  [context cleared]\n")
            continue
        if user == "/imgs":
            print(f"  attached: {len(pending_imgs)} image(s)\n")
            continue
        if user.startswith("/img"):
            path = user[4:].strip().strip('"')
            if not path:
                print("  usage: /img path/to/image.jpg\n")
                continue
            b64 = encode_image(path)
            if b64:
                pending_imgs.append(b64)
                print(f"  attached '{path}' ({len(b64) // 1024} KB base64) - it will be sent with your next message\n")
            continue

        msg = {"role": "user", "content": user}
        if pending_imgs:
            msg["images"] = pending_imgs
            pending_imgs = []
        history.append(msg)

        print("Bot: ", end="", flush=True)
        reply = stream_chat(base_url, history, headers)
        if reply is None:
            history.pop()          # failed turn: don't keep it in history
        else:
            history.append({"role": "assistant", "content": reply})

        # keep context manageable: vision tokens are heavy
        if len(history) > 24:
            history = history[-24:]
            print("  [trimmed oldest turns to stay within context]\n")


def main():
    parser = argparse.ArgumentParser(description="Report VLM timings from outputs or chat with the model.")
    parser.add_argument("--chat", action="store_true", help="Start the interactive Ollama chat client")
    parser.add_argument("base_url", nargs="?", help="Ollama URL (only used with --chat)")
    args = parser.parse_args()

    if args.chat:
        run_chat(args.base_url)
    elif args.base_url:
        parser.error("base_url can only be used with --chat")
    else:
        report_timings()


if __name__ == "__main__":
    main()
