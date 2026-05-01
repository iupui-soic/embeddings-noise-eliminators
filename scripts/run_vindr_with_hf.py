"""Wrapper that authenticates to HF before importing the main script."""
import os, sys, runpy
from huggingface_hub import login
tok_path = os.path.expanduser("~/.cache/huggingface/token")
with open(tok_path) as f:
    tok = f.read().strip()
login(token=tok, add_to_git_credential=False)
os.environ["HF_TOKEN"] = tok
runpy.run_path(
    "/home/saptpurk/embeddings-noise-eliminators/v4_work/compute_vindr_smallnodule.py",
    run_name="__main__")
