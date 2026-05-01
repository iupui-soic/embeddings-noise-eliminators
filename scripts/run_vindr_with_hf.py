"""Authenticate to HuggingFace, then papermill-execute notebook 21 (VinDr-CXR)."""
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import login

REPO_ROOT = Path("/home/saptpurk/embeddings-noise-eliminators")

tok_path = Path("~/.cache/huggingface/token").expanduser()
with tok_path.open() as f:
    tok = f.read().strip()
login(token=tok, add_to_git_credential=False)
os.environ["HF_TOKEN"] = tok

out_dir = REPO_ROOT / "outputs" / "papermill"
out_dir.mkdir(parents=True, exist_ok=True)

sys.exit(subprocess.call([
    "papermill",
    str(REPO_ROOT / "notebooks" / "21_VinDr_SmallNodule.ipynb"),
    str(out_dir / "21_VinDr_SmallNodule__vindr.ipynb"),
    "-p", "DATASET", "vindr",
    "-p", "MODELS", "raddino,dinov3",
    "-p", "REPO_ROOT_OVERRIDE", str(REPO_ROOT),
    "-p", "OUTPUTS_DIR", str(REPO_ROOT / "outputs"),
    "--log-output", "--log-level", "INFO",
]))
