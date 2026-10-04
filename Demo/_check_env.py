import os
print("HF_ENDPOINT:", repr(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_OFFLINE:", repr(os.environ.get("HF_HUB_OFFLINE")))

# Test actual download
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download("gpt2", "config.json")
    print("gpt2 config.json downloaded to:", path)
except Exception as e:
    print("Download failed:", e)
