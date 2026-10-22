"""Build a portable source + real research model bundle, without private runtime data.

Only the six explicitly listed public RSNA samples are initialized in a fresh DB.
The operational database, private uploads, .venv and 10 GB archive are never copied.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.database import initialize,transaction,event,now
from app.imaging import quality_check,validate_png


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('/workspace/deliverables/BoneAtlas_MVP_0.1.zip'))
    ap.add_argument('--samples',type=Path,default=Path('/workspace/rsna-image-review'))
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'var/models/active.json').read_text())
    checkpoint=root/'var/models'/Path(meta['checkpoint']).name
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==meta['checkpoint_sha256']
    with (root/'boneage-training-dataset.csv').open() as f:labels={r['id']:r for r in csv.DictReader(f)}
    expected={r['file'].removeprefix('train').removesuffix('.png'):r['sha256']
              for r in json.loads((root/'docs/reports/sample-manifest.json').read_text())
              if r['file'].startswith('train')}
    base='bone-atlas/'
    with tempfile.TemporaryDirectory(prefix='bone-package-') as tmp:
        database=Path(tmp)/'workbench.sqlite3';initialize(database)
        sample_files=[]
        with transaction(database) as db:
            for source_id in ('1377','1378','3806','6197','11929','14466'):
                path=args.samples/f'train{source_id}.png'
                if not path.is_file():path=root/'var/images'/f'{expected[source_id]}.png'
                raw=path.read_bytes();image=validate_png(raw);sha=hashlib.sha256(raw).hexdigest()
                if sha!=expected[source_id]:raise RuntimeError('Unexpected public sample checksum')
                flags,metrics=quality_check(image);case_id=uuid.uuid4().hex
                sex='M' if labels[source_id]['male']=='True' else 'F'
                timestamp=now()
                db.execute('''INSERT INTO cases(id,source_image_id,filename,image_sha256,sex,sex_source,width,height,
                    qc_status,qc_flags,qc_metrics,created_at,updated_at) VALUES(?,?,?,?,?,'csv',?,?,'pending',?,?,?,?)''',
                    (case_id,source_id,f'{source_id}.png',sha,sex,image.width,image.height,
                     json.dumps(flags,ensure_ascii=False),json.dumps(metrics),timestamp,timestamp))
                event(db,case_id,'public_sample_import',{'dataset':'RSNA Bone Age v2','source_image_id':source_id,
                                                        'sex_source':'supplied training CSV','sha256':sha})
                sample_files.append((path,f'var/images/{sha}.png'))
        with sqlite3.connect(database) as db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        with ZipFile(args.output,'w',compression=ZIP_DEFLATED,compresslevel=6) as out:
            for name in ('README.md','.gitignore','requirements.txt','requirements-ml.txt','requirements-dev.txt','requirements.lock',
                         'boneage-training-dataset.csv','boneage-test-dataset.csv'):
                out.write(root/name,base+name)
            for folder in ('app','scripts','tests','docs','.github'):
                for path in sorted((root/folder).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                        out.write(path,base+str(path.relative_to(root)))
            out.write(database,base+'var/workbench.sqlite3')
            for path,name in sample_files:out.write(path,base+name)
            for name in ('active.json','evaluation.json',checkpoint.name,'test-predictions.csv'):
                out.write(root/'var/models'/name,base+'var/models/'+name)
            out.write(root/'var/unlabeled-predictions.csv',base+'var/unlabeled-predictions.csv')
            for name in ('audit.json','download.json'):
                path=Path('/workspace/bone-data')/name
                if path.exists():out.write(path,base+'data/'+name)
            out.writestr(base+'PACKAGE.json',json.dumps({
                'version':'0.1.0','scope':'single-image research prototype','public_samples':6,
                'operational_database_included':False,'private_uploads_included':False,
                'model_id':meta['model_id'],'model_sha256':meta['checkpoint_sha256'],
                'test_mae_months':meta['evaluation']['test']['mae_months'],
                'target_passed':meta['evaluation']['target_passed'],
                'raw_full_dataset_included':False,
            },ensure_ascii=False,indent=2))
        with ZipFile(args.output) as out:
            bad=out.testzip()
            if bad:raise RuntimeError('Package CRC failed: '+bad)
            print('Verified',len(out.infolist()),'archive entries')
    sha=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(f'{sha}  {args.output.name}\n')
    print(args.output,'bytes',args.output.stat().st_size,'sha256',sha)


if __name__=='__main__':main()
