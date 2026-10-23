"""Download the public, versioned Kaggle archive using verified HTTPS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Settings.from_env().dataset)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'rsna-bone-age-v2.zip'
    if target.exists():
        with ZipFile(target) as archive:
            count = len(archive.infolist())
        print(f'Existing archive retained ({count} entries); run prepare_data.py for full CRC audit.')
        return
    url = 'https://www.kaggle.com/api/v1/datasets/download/kmader/rsna-bone-age?datasetVersionNumber=2'
    partial = target.with_suffix('.zip.part')
    started = last = time.monotonic()
    total = 0
    with urlopen(Request(url, headers={'User-Agent': 'BoneAge-Research/0.1'}), timeout=60) as response:
        if response.headers.get('Content-Type', '').split(';')[0] not in {'application/zip', 'application/octet-stream'}:
            raise RuntimeError('Kaggle did not return an archive. Check access and dataset terms.')
        expected = int(response.headers.get('Content-Length', '0'))
        with partial.open('wb') as output:
            while chunk := response.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > 12 * 1024**3:
                    raise RuntimeError('Download exceeded 12 GiB safety limit')
                output.write(chunk)
                if time.monotonic() - last > 20:
                    print(f'{total / 1024**3:.2f} GiB / {expected / 1024**3:.2f} GiB', flush=True)
                    last = time.monotonic()
    if expected and total != expected:
        raise RuntimeError('Incomplete download; partial file retained for diagnosis')
    with ZipFile(partial) as archive:
        if 'boneage-training-dataset.csv' not in archive.namelist():
            raise RuntimeError('Unexpected archive contents')
    os.replace(partial, target)
    with target.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    (args.output / 'download.json').write_text(json.dumps({
        'source': url, 'dataset_version': 2, 'bytes': total, 'sha256': sha,
        'elapsed_seconds': round(time.monotonic()-started),
    }, indent=2))
    print(f'Saved {target}; sha256={sha}')


if __name__ == '__main__':
    main()
