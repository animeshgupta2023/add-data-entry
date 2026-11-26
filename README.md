# AI-Powered Advertisement Data Entry

Extract structured information from newspaper advertisement images with a
vision-language model (VLM) served by Ollama. The recommended setup runs Ollama
in the Kaggle notebook at [`kaggle/add-data-entry.ipynb`](kaggle/add-data-entry.ipynb)
and runs the batch client from this repository, sending image requests to
Kaggle through a temporary Cloudflare tunnel.

The proof of concept uses 50 multilingual Indian newspaper advertisements. It
supports batch processing, saved progress, per-ad timing, and model timing
reports.

## Requirements

- A Kaggle account with Internet and a GPU accelerator enabled for the notebook
- Python 3.9 or newer on the computer running this repository
- The Python dependencies in [`requirements.txt`](requirements.txt)

The notebook installs and runs Ollama on Kaggle and downloads both supported
model tags. Ollama does not need to be installed on the computer running the
Python client.

## 1. Start the Kaggle Ollama server

1. Import or open `kaggle/add-data-entry.ipynb` as a Kaggle notebook.
2. In the notebook settings, enable Internet and select a GPU accelerator.
3. Run all cells. The notebook installs Ollama and `cloudflared`, starts the
   Ollama server, pulls `qwen3-vl:8b` and `qwen2.5vl:7b`, and starts a Cloudflare
   quick tunnel to the Ollama API.
4. Wait for the final setup cell to print `Your Ollama endpoint:`, then copy
   the full `https://....trycloudflare.com` URL.
5. Keep the Kaggle notebook session running while you use the endpoint.

The tunnel URL is temporary and changes when the Kaggle session or tunnel is
restarted. Copy the new URL and update `OLLAMA_HOST` whenever that happens.

> **Tunnel warning:** A Cloudflare quick tunnel publicly exposes the Ollama
> API to anyone who has its URL. This notebook is intended for short-lived
> testing. Do not send sensitive images or data through it, and stop the Kaggle
> session when finished.

## 2. Set up the local Python client

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the endpoint copied from the Kaggle notebook in the same PowerShell
session:

```powershell
$env:OLLAMA_HOST = "https://your-current-tunnel.trycloudflare.com"
```

Replace the example URL with the current URL printed by the notebook. The
environment variable is session-scoped; set it again in a new terminal.
Alternatively, pass the endpoint directly with `--host`.

## 3. Run batch extraction

Run commands from the repository root. By default, the client:

- Reads images from `dataset\images`.
- Uses `qwen3-vl:8b`.
- Saves results to `outputs\extracted_ads_qwen3-vl_8b.json`.
- Resumes from successful entries already saved in that output file.

Run the default model:

```powershell
python src\cli.py
```

Run a small test using a separate output file:

```powershell
python src\cli.py --limit 3 --out outputs\qwen3-smoke-test.json
```

Run the other model; it gets a separate default output file:

```powershell
python src\cli.py --model qwen2.5vl:7b
```

Pass the endpoint directly instead of setting `OLLAMA_HOST`:

```powershell
python src\cli.py --host https://your-current-tunnel.trycloudflare.com
```

### CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `qwen3-vl:8b` | Ollama model tag to run. |
| `--input` | `dataset\images` | Directory of images or a path to one image. Supported extensions: JPG, JPEG, PNG, and WebP. |
| `--out` | `outputs\extracted_ads_<model>.json` | Output JSON path. |
| `--host` | `OLLAMA_HOST` if set; otherwise a built-in fallback | Ollama server URL. Set this explicitly to the current Kaggle tunnel URL. |
| `--max-dim` | `1536` | Maximum image dimension for the first extraction attempt. |
| `--limit` | All images | Process at most the first N sorted images. |
| `--force` | Off | Ignore saved progress and process images again. |

When an output file already exists, successful entries are skipped; failed or
unfinished entries are retried. Use `--force` to rerun all images, or use a
different `--out` path to preserve the existing results. A smoke test should
use a separate output file if a previous full run already exists.

The client sends each image to the Kaggle Ollama API and writes its result
locally. Keep the Kaggle notebook and tunnel alive for the entire run. Results
are saved after each processed image, so successful work can be resumed after
an interruption.

## Output format

Each output file contains the model name, input directory, processed/success/
error counts, average VLM time per successful ad, and an `ads` object keyed by
image ID.

Each successful ad entry contains its filename, status, attempt number,
`vlm_time_sec`, and extracted `data`. The extracted data includes:

- Category, brand/company, product/service, and headline
- Pricing and offers, including details, amount, and conditions
- Key features and category-specific details
- Shops/dealers and contact information
- Detected languages and fields that could not be confidently read

Failed entries have `status: "error"` and an error message instead of extracted
data. Model-generated fields should be reviewed, especially small text, prices,
contact details, and regional-language content.

## Timing report and chat utility

After generating output files, run the timing report:

```powershell
python src\test.py
```

It sums `vlm_time_sec` for successful ads in each
`outputs\extracted_ads_*.json` file and reports per-model and combined totals.
This is the accumulated VLM time, not the wall-clock duration of the entire
batch. The batch client separately prints its elapsed run time when it finishes.

The same script has an optional interactive chat mode. It uses `OLLAMA_URL`
when set, or accepts the server URL as an argument:

```powershell
python src\test.py --chat https://your-current-tunnel.trycloudflare.com
```

In chat mode, type `/img <path>` to attach an image, `/reset` to clear the
conversation, `/imgs` to see the number of queued images, or `/quit` to exit.
The Kaggle notebook and tunnel must remain running for chat requests too.

## Dataset and attribution

The sample images and metadata are in [`dataset/`](dataset/). The collection
contains 50 cropped newspaper advertisements, including English and
regional-language examples. Dataset sourcing, attribution, and usage
information are documented in [`dataset/README.md`](dataset/README.md). The
images are intended for non-commercial research and testing; respect the
original publishers' terms.

## Current scope

The pipeline sends images directly to the VLM; PaddleOCR preprocessing is not
used. Extraction quality and runtime depend on the selected model, image
quality, Kaggle GPU availability, and network/tunnel stability.
