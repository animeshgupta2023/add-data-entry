paddleOCR can only do 
- text detection 
- text recognition
- text direction classification


# 1. Quick sanity test on the first 3 images
python src/cli.py --limit 3

# 2. Run on the entire dataset/images folder (resumes automatically if interrupted)
python src/cli.py

# 3. Benchmark Qwen2.5-VL-7B into a separate single JSON file
python src/cli.py --model qwen2.5vl:7b --out outputs\extracted_ads_qwen25vl.json


skippign the paddle ocr as this causes the vlm's performance to go down for odia or such languages and causes confusion 
