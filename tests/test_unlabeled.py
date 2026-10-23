import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
import torch

from app.network import BoneAgeCNN
from scripts.predict_unlabeled import predict_archive


class UnlabeledExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.models=self.root/'models';self.models.mkdir()
        self.model=BoneAgeCNN()
        with torch.no_grad():
            for parameter in self.model.parameters():parameter.zero_()
        self.meta={'model_id':'synthetic-model','architecture':self.model.architecture,
                   'checkpoint':'model.pt','image_size':96,'preprocess_version':'gray-letterbox-v1',
                   'evaluation':{'target_passed':False}}
        self.save_model()
        self.table=self.root/'unlabeled.csv'
        self.table.write_text('Case ID,Sex\n1,F\n2,M\n3,F\n',encoding='utf-8')
        self.archive=self.root/'images.zip'
        buffer=io.BytesIO();Image.new('L',(100,160),100).save(buffer,format='PNG')
        self.png=buffer.getvalue()
        with ZipFile(self.archive,'w') as archive:
            for source_id in ('1','2','3'):
                archive.writestr(f'boneage-test-dataset/boneage-test-dataset/{source_id}.png',self.png)
        self.output=self.root/'output.csv'

    def tearDown(self):
        self.temp.cleanup()

    def save_model(self):
        checkpoint=self.models/'model.pt';torch.save(self.model.state_dict(),checkpoint)
        self.meta['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (self.models/'active.json').write_text(json.dumps(self.meta),encoding='utf-8')

    def run_export(self):
        return predict_archive(self.archive,self.models,self.output,batch_size=2,table=self.table)

    def test_all_ids_and_tail_batch_are_exported_with_model_identity_and_no_accuracy(self):
        report=self.run_export()
        with self.output.open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual([r['Case ID'] for r in rows],['1','2','3'])
        self.assertEqual([r['Sex'] for r in rows],['F','M','F'])
        self.assertTrue(all(float(r['predicted_boneage_months'])==120 for r in rows))
        self.assertEqual(report['valid_rows'],3)
        self.assertFalse(report['mae_computed'])
        self.assertFalse(report['ground_truth_available'])
        self.assertEqual(report['predictions_sha256'],hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_negative_predictions_preserve_rows_and_are_marked_invalid(self):
        with torch.no_grad():self.model.head[-1].bias.fill_(-3)
        self.save_model();report=self.run_export()
        with self.output.open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(report['invalid_rows'],3)
        self.assertTrue(all(r['status']=='invalid_output' and r['predicted_boneage_months']=='' for r in rows))

    def test_corrupt_weights_and_model_changes_preserve_previous_export(self):
        self.output.write_text('previous export',encoding='utf-8')
        with (self.models/'model.pt').open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_export()
        self.assertEqual(self.output.read_text(),'previous export')
        self.save_model()
        with patch('scripts.predict_unlabeled.Predictor.metadata',side_effect=[self.meta,{**self.meta,'model_id':'changed'}]):
            with self.assertRaisesRegex(RuntimeError,'changed'):self.run_export()
        self.assertEqual(self.output.read_text(),'previous export')
        self.assertEqual(list(self.root.glob('.unlabeled-*')),[])

    def test_duplicate_ids_and_audited_image_mismatch_are_refused(self):
        self.table.write_text('Case ID,Sex\n1,F\n1,M\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_export()
        self.table.write_text('Case ID,Sex\n1,F\n2,M\n3,F\n',encoding='utf-8')
        rows=[{'kind':'unlabeled','status':'valid','source_id':i,'sex':sex,
               'archive_path':f'boneage-test-dataset/boneage-test-dataset/{i}.png','sha256':'0'*64}
              for i,sex in (('1','F'),('2','M'),('3','F'))]
        raw='\n'.join(json.dumps(r) for r in rows)+'\n'
        (self.root/'manifest.jsonl').write_text(raw,encoding='utf-8')
        (self.root/'audit.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(raw.encode()).hexdigest()}))
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_export()
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
