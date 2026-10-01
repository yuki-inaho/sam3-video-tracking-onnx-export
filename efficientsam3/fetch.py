"""Fetch the pinned public upstream source and trained EV-M checkpoint."""

import argparse
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from contract import HF_FILE, HF_REPO, HF_REVISION, SOURCE_REVISION, SOURCE_URL, verify_file


def fetch_source(destination):
    destination = Path(destination)
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", SOURCE_REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Source revision mismatch: {revision}")
    if subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"]):
        raise ValueError("Upstream source checkout has local modifications")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/efficientsam3/source")
    p.add_argument("--checkpoint", default="models/efficientsam3_ev_m.pt")
    a = p.parse_args()
    fetch_source(a.source)
    target = Path(a.checkpoint)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        # Cache busting avoids stale, expired signed CDN redirects from proxies.
        url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"
        try:
            with urllib.request.urlopen(f"{url}?download=true&nocache={time.time_ns()}") as r:
                with temporary.open("wb") as f:
                    shutil.copyfileobj(r, f)
            verify_file(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"EV-M SHA256 verified: {verify_file(target)}")


if __name__ == "__main__":
    main()
