"""Import real public PNG examples, without fabricated results or automatic QC acceptance."""
import argparse
import csv
import io
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import httpx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--samples',type=Path,default=Path('/workspace/rsna-image-review'))
    ap.add_argument('--archive',type=Path,default=Path('/workspace/bone-data/rsna-bone-age-v2.zip'))
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    with (root/'boneage-training-dataset.csv').open() as f:labels={r['id']:r for r in csv.DictReader(f)}
    manifest_file=root/'docs/reports/sample-manifest.json'
    sample_hashes={r['file'].removeprefix('train').removesuffix('.png'):r['sha256']
                   for r in json.loads(manifest_file.read_text()) if r['file'].startswith('train')}
    archive=None
    try:
        with httpx.Client(base_url=args.url,timeout=40) as client:
            for source_id in ('1377','1378','3806','6197','11929','14466'):
                path=args.samples/f'train{source_id}.png'
                if not path.is_file():
                    path=root/'var/images'/f'{sample_hashes[source_id]}.png'
                if path.is_file(): data=path.read_bytes()
                elif args.archive.exists():
                    archive=archive or ZipFile(args.archive)
                    data=archive.read(f'boneage-training-dataset/boneage-training-dataset/{source_id}.png')
                else: raise RuntimeError('No sample PNG or original archive available; download data first')
                if hashlib.sha256(data).hexdigest()!=sample_hashes[source_id]:
                    raise RuntimeError(f'Public sample checksum mismatch: {source_id}')
                sex='M' if labels[source_id]['male']=='True' else 'F'
                response=client.post('/api/cases',files={'file':(f'{source_id}.png',data,'image/png')},
                    data={'sex':sex,'source_image_id':source_id,'sex_source':'csv'})
                if response.status_code not in (201,409):response.raise_for_status()
                print(source_id,'imported' if response.status_code==201 else 'already present')
    finally:
        if archive: archive.close()


if __name__=='__main__':main()
