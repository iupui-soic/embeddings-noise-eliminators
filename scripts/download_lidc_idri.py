#!/usr/bin/env python3
"""
Download LIDC-IDRI from TCIA into a flat per-patient layout under DEST.

Layout:
  DEST/<PatientID>/<Modality>/<SeriesInstanceUID>/<*.dcm>

Resumable: a series whose target directory already contains a non-empty
DICOM file is skipped. Per-series failures are recorded in MANIFEST_CSV
and the run continues. Re-running the script picks up where it left off.

Modality priority order (CT first, derived data last) so the imaging is
on disk early even if the run is interrupted:

  CT  ->  DX  ->  CR  ->  SEG  ->  SR

Usage:
  python scripts/download_lidc_idri.py                   # full collection
  python scripts/download_lidc_idri.py --max-series 1    # smoke test
  python scripts/download_lidc_idri.py --modalities CT   # subset
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

from tcia_utils import nbia

DEST = Path(os.environ.get("LIDC_DEST", "/data0/lidc-idri"))
LOG_PATH = Path(os.environ.get(
    "LIDC_LOG", "/home/saptpurk/embeddings-noise-eliminators/outputs/lidc_download.log"))
MANIFEST_PATH = Path(os.environ.get(
    "LIDC_MANIFEST",
    "/home/saptpurk/embeddings-noise-eliminators/outputs/lidc_download_manifest.csv"))

PRIORITY = ["CT", "DX", "CR", "SEG", "SR"]


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lidc")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # Quiet tcia_utils' own root-logger noise but keep WARN+
    logging.getLogger("tcia_utils.nbia").setLevel(logging.WARNING)
    return logger


def already_downloaded(target: Path) -> bool:
    if not target.exists():
        return False
    for entry in target.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".dcm" and entry.stat().st_size > 0:
            return True
    return False


def load_manifest() -> dict[str, str]:
    """Return {series_uid: status} from prior runs."""
    if not MANIFEST_PATH.exists():
        return {}
    out: dict[str, str] = {}
    with MANIFEST_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["series_uid"]] = row.get("status", "")
    return out


def append_manifest(row: dict) -> None:
    write_header = not MANIFEST_PATH.exists()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts", "series_uid", "patient_id", "modality",
                "image_count", "filesize", "status", "elapsed_s", "error",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def download_one(series: dict, dest_root: Path, logger: logging.Logger) -> tuple[str, float, str]:
    suid = series["SeriesInstanceUID"]
    pid = series.get("PatientID", "UNKNOWN")
    mod = series.get("Modality", "UNK")

    target_dir = dest_root / pid / mod / suid
    if already_downloaded(target_dir):
        return ("skipped", 0.0, "")

    # Only create the parent (Modality) directory; tcia_utils will create the
    # SeriesInstanceUID subfolder itself, and skips if that subfolder pre-exists.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        # tcia_utils with input_type="list" wants a list of UID strings.
        nbia.downloadSeries(
            series_data=[suid],
            path=str(target_dir.parent),
            input_type="list",
            api_url="",
        )
        elapsed = time.time() - t0
        # tcia_utils writes into <path>/<SeriesInstanceUID>/, which is exactly target_dir.
        if not already_downloaded(target_dir):
            return ("failed", elapsed, "no .dcm files written")
        return ("ok", elapsed, "")
    except Exception as exc:
        elapsed = time.time() - t0
        logger.warning(f"download failed for {suid}: {exc}")
        return ("failed", elapsed, str(exc)[:300])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-series", type=int, default=None,
                   help="Stop after this many series (smoke testing).")
    p.add_argument("--modalities", nargs="+", default=PRIORITY,
                   help="Modality whitelist; default downloads all in priority order.")
    p.add_argument("--collection", default="LIDC-IDRI")
    args = p.parse_args()

    logger = setup_logging()
    DEST.mkdir(parents=True, exist_ok=True)

    logger.info(f"DEST={DEST}  LOG={LOG_PATH}  MANIFEST={MANIFEST_PATH}")
    logger.info(f"Listing series for collection={args.collection} ...")
    series = nbia.getSeries(collection=args.collection, api_url="")
    logger.info(f"Total series returned by API: {len(series)}")

    by_mod: dict[str, list[dict]] = {m: [] for m in PRIORITY}
    for s in series:
        m = s.get("Modality", "UNK")
        if m in args.modalities:
            by_mod.setdefault(m, []).append(s)

    ordered: list[dict] = []
    for m in PRIORITY:
        if m in args.modalities:
            ordered.extend(by_mod.get(m, []))

    if args.max_series is not None:
        ordered = ordered[: args.max_series]

    logger.info(
        "Plan: " + ", ".join(f"{m}={len(by_mod.get(m, []))}" for m in PRIORITY)
        + f"  -> total to process: {len(ordered)}"
    )

    prior = load_manifest()
    n_total = len(ordered)
    n_ok = n_skip = n_fail = 0

    for i, s in enumerate(ordered, 1):
        suid = s["SeriesInstanceUID"]
        if prior.get(suid) == "ok":
            n_skip += 1
            if i % 200 == 0:
                logger.info(f"[{i}/{n_total}] checkpoint skip; ok={n_ok} skip={n_skip} fail={n_fail}")
            continue

        status, elapsed, err = download_one(s, DEST, logger)
        append_manifest({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "series_uid": suid,
            "patient_id": s.get("PatientID"),
            "modality": s.get("Modality"),
            "image_count": s.get("ImageCount"),
            "filesize": s.get("FileSize"),
            "status": status,
            "elapsed_s": f"{elapsed:.1f}",
            "error": err,
        })

        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_fail += 1

        if i % 25 == 0 or i == n_total:
            logger.info(
                f"[{i}/{n_total}] {s.get('PatientID')} {s.get('Modality')} "
                f"{status} ({elapsed:.1f}s) | running: ok={n_ok} skip={n_skip} fail={n_fail}"
            )

    logger.info(f"DONE. ok={n_ok} skip={n_skip} fail={n_fail} total={n_total}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
