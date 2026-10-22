"""Write predictions for the 200 unlabeled RSNA images; never report an accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.imaging import letterbox, validate_png
from app.network import BoneAgeCNN, denormalize_target, normalize_image


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=Path('/workspace/bone-data/rsna-bone-age-v2.zip'))
    ap.add_argument('--models',type=Path,default=root/'var/models')
    ap.add_argument('--output',type=Path,default=root/'var/unlabeled-predictions.csv')
    args=ap.parse_args()
    meta=json.loads((args.models/'active.json').read_text())
    checkpoint=args.models/Path(meta['checkpoint']).name
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
        raise RuntimeError('Checkpoint checksum mismatch')
    model=BoneAgeCNN();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    torch.set_num_threads(2)
    with (root/'boneage-test-dataset.csv').open() as f:rows=list(csv.DictReader(f))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.csv.tmp')
    with ZipFile(args.archive) as archive, temporary.open('w',newline='') as f, torch.no_grad():
        writer=csv.writer(f);writer.writerow(['Case ID','Sex','predicted_boneage_months','model_id','status'])
        for row in rows:
            data=archive.read(f"boneage-test-dataset/boneage-test-dataset/{row['Case ID']}.png")
            image=validate_png(data);array,_=letterbox(image,meta['image_size'])
            x=normalize_image(torch.from_numpy(array).unsqueeze(0).unsqueeze(0))
            sex=torch.tensor([[1.0 if row['Sex']=='M' else 0.0]])
            value=float(denormalize_target(model(x,sex)).item())
            valid=np.isfinite(value) and 0<=value<=360
            writer.writerow([row['Case ID'],row['Sex'],round(value,4) if valid else '',meta['model_id'],
                             'research_prediction' if valid else 'invalid_output'])
    temporary.replace(args.output)
    print(f'{len(rows)} unlabeled predictions saved to {args.output}. No ground truth exists here; no MAE was computed.')


if __name__=='__main__':main()
