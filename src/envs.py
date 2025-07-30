import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------


HF_TOKEN = os.environ.get("HF_TOKEN")
SAHARA_DATA = os.environ.get("SAHARA_DATA")
SAHARA_RESULTS = os.environ.get("SAHARA_RESULTS")
API = HfApi(token=HF_TOKEN)
