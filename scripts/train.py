"""Train a real CPU CNN baseline; hold-out test is used only after model selection."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import now
from app.imaging import PREPROCESS_VERSION
from app.network import BoneAgeCNN, denormalize_target, normalize_image


def metrics(predictions, labels, sex):
    errors = np.abs(predictions - labels)
    def subset(mask):
        e = errors[mask]
        return {'n': int(mask.sum()), 'mae_months': round(float(e.mean()), 4) if len(e) else None}
    result = {'n': len(labels), 'mae_months': round(float(errors.mean()), 4),
              'rmse_months': round(float(np.sqrt(((predictions-labels)**2).mean())), 4),
              'median_absolute_error': round(float(np.median(errors)), 4),
              'p90_absolute_error': round(float(np.percentile(errors, 90)), 4),
              'by_sex': {'M': subset(sex == 1), 'F': subset(sex == 0)}, 'by_boneage': {}}
    for low, high in ((0,24),(24,72),(72,120),(120,168),(168,1000)):
        result['by_boneage'][f'{low}-{high if high<1000 else "+"}'] = subset((labels>=low)&(labels<high))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('/workspace/bone-data'))
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'var/models')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=96)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261022)
    args = ap.parse_args()
    if args.epochs < 1: ap.error('epochs must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    audit = json.loads((args.data/'audit.json').read_text())
    size = audit['image_size']
    cache_file = args.data/f'cache-{size}.npz'
    if hashlib.sha256(cache_file.read_bytes()).hexdigest() != audit['cache_sha256']:
        raise RuntimeError('Cache checksum mismatch; regenerate audit/cache')
    cache = np.load(cache_file, allow_pickle=False)
    x=torch.from_numpy(cache['images']).unsqueeze(1)
    sex=torch.from_numpy(cache['sex']).unsqueeze(1)
    y=torch.from_numpy(cache['targets'])
    split=cache['split']
    indices = {name:np.flatnonzero(split==code) for code,name in enumerate(('train','validation','test'))}
    if min(map(len,indices.values())) < 10: raise RuntimeError('Insufficient independent split samples')
    datasets = {k:TensorDataset(x[v],sex[v],y[v]) for k,v in indices.items()}
    loaders = {k:DataLoader(v,batch_size=args.batch_size,shuffle=(k=='train'),num_workers=0)
               for k,v in datasets.items()}
    model=BoneAgeCNN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0015,weight_decay=0.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=0.00005)
    loss_function=nn.SmoothL1Loss(beta=.5)
    def predict(loader):
        model.eval();pred=[];truth=[];sexes=[]
        with torch.no_grad():
            for bx,bs,by in loader:
                pred.extend(denormalize_target(model(normalize_image(bx),bs)).tolist())
                truth.extend(by.tolist());sexes.extend(bs[:,0].tolist())
        return np.asarray(pred),np.asarray(truth),np.asarray(sexes)
    args.output.mkdir(parents=True,exist_ok=True)
    progress_file=args.output/'training-progress.json'
    start=time.monotonic();history=[];best=float('inf');best_state=None;best_epoch=0
    for epoch in range(args.epochs):
        model.train();loss_total=0;n=0
        for bx,bs,by in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            bx=normalize_image(bx)
            # Mild intensity augmentation only on train; no mirroring or anatomy-distorting crop.
            contrast=torch.empty((len(bx),1,1,1)).uniform_(.9,1.1)
            brightness=torch.empty((len(bx),1,1,1)).uniform_(-.06,.06)
            bx=(bx*contrast+brightness).clamp(-1,1)
            output=model(bx,bs)
            loss=loss_function(output,(by-120)/60)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();loss_total+=float(loss.detach())*len(bx);n+=len(bx)
        schedule.step()
        vp,vy,vs=predict(loaders['validation'])
        mae=float(np.abs(vp-vy).mean())
        if mae<best:
            best=mae;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        record={'epoch':epoch+1,'train_loss':round(loss_total/n,5),
                'validation_mae_months':round(mae,4),'elapsed_seconds':round(time.monotonic()-start)}
        history.append(record)
        progress_file.write_text(json.dumps({'status':'training','total_epochs':args.epochs,'history':history,'best_epoch':best_epoch}))
        print(json.dumps(record),flush=True)
    model.load_state_dict(best_state)
    # One final test after selection by validation MAE, never used to choose epoch.
    tp,ty,ts=predict(loaders['test'])
    result=metrics(tp,ty,ts)
    report={'created_at':now(),'architecture':BoneAgeCNN.architecture,'training_samples':len(indices['train']),
            'validation_samples':len(indices['validation']),'test_samples':len(indices['test']),
            'epochs':args.epochs,'best_epoch':best_epoch,'best_validation_mae_months':round(best,4),
            'test':result,'target_mae_months':6,'target_passed':result['mae_months']<6,
            'patient_isolation':audit['patient_isolation'], 'grouping':audit['grouping'],
            'label_reference_standard':'Not provided; public RSNA labels, not validated against Chinese clinical standard',
            'manifest_sha256':audit['manifest_sha256'],'cache_sha256':audit['cache_sha256'],
            'preprocess_version':PREPROCESS_VERSION,'image_size':size,
            'seed':args.seed,'runtime':{'torch':torch.__version__,'device':'cpu','threads':args.threads},
            'limitations':['Research CPU baseline trained from scratch, not a clinical model',
                           'Patient independence is unverified; no patient identifiers supplied',
                           'Near-duplicate search is not exhaustive; no external clinical validation',
                           '8-bit grayscale PNG only; no DICOM, ROI labels or longitudinal outcomes'],
            'history':history,'elapsed_seconds':round(time.monotonic()-start)}
    model_id='rsna-cpu-cnn-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    checkpoint=args.output/(model_id+'.pt')
    torch.save(best_state,checkpoint)
    meta={'model_id':model_id,'architecture':BoneAgeCNN.architecture,'checkpoint':checkpoint.name,
          'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'image_size':size,
          'preprocess_version':PREPROCESS_VERSION,'created_at':now(),'usage':'research_only',
          'explanation':'Grad-CAM of predicted boneage; relative attention, not a diagnostic segmentation',
          'evaluation':{k:report[k] for k in ('training_samples','validation_samples','test_samples','best_epoch',
                         'best_validation_mae_months','test','target_mae_months','target_passed',
                         'patient_isolation','limitations')}}
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Atomic registration only after checkpoint and evaluation are complete.
    temporary=args.output/'active.json.tmp';temporary.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    os.replace(temporary,args.output/'active.json')
    progress_file.write_text(json.dumps({'status':'finished','total_epochs':args.epochs,'history':history,'best_epoch':best_epoch}))
    with (args.output/'test-predictions.csv').open('w',newline='') as f:
        import csv
        writer=csv.writer(f);writer.writerow(['source_image_id','reference_months','predicted_months','sex'])
        for i,pred,truth,s in zip(indices['test'],tp,ty,ts):writer.writerow([cache['ids'][i],truth,round(float(pred),4),'M' if s else 'F'])
    print(json.dumps({'model_id':model_id,'test':result,'target_passed':report['target_passed'],
                      'best_epoch':best_epoch,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
