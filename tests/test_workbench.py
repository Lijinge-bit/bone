import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.database import transaction
from app.imaging import ImageValidationError, letterbox, validate_png
from app.main import create_app


def png(size=(128,192),mode='L',value=80):
    b=io.BytesIO();Image.new(mode,size,value).save(b,format='PNG');return b.getvalue()


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name)
        self.settings=Settings(root/'var',root/'data')
        self.app=create_app(self.settings)
        self.context=TestClient(self.app)
        self.client=self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None,None,None)
        self.temp.cleanup()

    def upload(self,data=None,sex='F'):
        return self.client.post('/api/cases',files={'file':('1377.png',data or png(),'image/png')},
                                data={'sex':sex,'source_image_id':'1377'})

    def ready(self):
        c=self.upload().json()
        r=self.client.post(f"/api/cases/{c['id']}/quality",json={'revision':c['revision'],
            'decision':'usable','reviewer':'test reviewer','note':'目标手部已核对，允许研究评估'})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def model_meta(self):
        return {'model_id':'unit-test-model','checkpoint_sha256':'checked-in-test',
                'preprocess_version':'gray-letterbox-v1'}

    def assess_test_double(self,c,result=None):
        # This double tests task persistence, not model accuracy. Real model is checked separately.
        if result is None:result=(123.5,Image.new('RGBA',(128,192),(255,0,0,80)),None)
        with patch.object(self.app.state.predictor,'metadata',return_value=self.model_meta()), \
             patch.object(self.app.state.predictor,'assess',return_value=result):
            r=self.client.post(f"/api/cases/{c['id']}/assess",json={'revision':c['revision']})
            self.assertEqual(r.status_code,202,r.text)
            for _ in range(100):
                current=self.client.get(f"/api/cases/{c['id']}").json()
                if current['evaluations'][0]['status'] not in {'queued','running'}:return current
                time.sleep(.01)
        self.fail('background task did not finish')

    def test_import_duplicate_and_identity(self):
        r=self.upload();self.assertEqual(r.status_code,201)
        c=r.json();self.assertEqual(c['qc_status'],'pending')
        self.assertEqual(self.upload().status_code,409)
        self.assertEqual(self.client.get(c['image_url']).content,png())
        self.assertNotIn('patient_id',c)
        self.assertEqual(self.client.get('/api/status').json()['stats']['total'],1)

    def test_invalid_format_color_and_missing_sex(self):
        self.assertEqual(self.upload(b'not a PNG').status_code,422)
        self.assertEqual(self.upload(png(mode='RGB',value=(80,80,80))).status_code,422)
        self.assertEqual(self.upload(sex='False').status_code,422)
        self.assertIsNone(self.upload(sex='').json()['sex'])

    def test_quality_gate_and_missing_model(self):
        c=self.upload().json()
        with patch.object(self.app.state.predictor,'metadata',return_value=self.model_meta()):
            r=self.client.post(f"/api/cases/{c['id']}/assess",json={'revision':c['revision']})
            self.assertEqual(r.status_code,422)
        c=self.client.get(f"/api/cases/{c['id']}").json()
        r=self.client.post(f"/api/cases/{c['id']}/assess",json={'revision':c['revision']})
        self.assertEqual(r.status_code,503)
        self.assertEqual(self.client.get(f"/api/cases/{c['id']}").json()['evaluations'],[])

    def test_quality_reject_and_optimistic_revision(self):
        c=self.ready()
        r=self.client.patch(f"/api/cases/{c['id']}/sex",json={'revision':1,'sex':'M'})
        self.assertEqual(r.status_code,409)
        r=self.client.post(f"/api/cases/{c['id']}/quality",json={'revision':c['revision'],
            'decision':'reject','reviewer':'reviewer','note':'目标不明确，需重新选择'})
        self.assertEqual(r.json()['qc_status'],'rejected')

    def test_actual_snapshot_review_and_staleness(self):
        c=self.ready();detail=self.assess_test_double(c);e=detail['evaluations'][0]
        self.assertEqual(e['prediction_months'],123.5)
        self.assertEqual(e['input_sex'],'F')
        self.assertEqual(self.client.get(e['heatmap_url']).status_code,200)
        r=self.client.post(f"/api/evaluations/{e['id']}/reviews",json={
            'reviewer':'医师 A','decision':'question','note':'保留原始输出，进一步核对','confirmed_months':120})
        self.assertEqual(r.status_code,201)
        revised=self.client.patch(f"/api/cases/{c['id']}/sex",json={'revision':c['revision'],'sex':'M'}).json()
        detail=self.client.get(f"/api/cases/{c['id']}").json()
        old=detail['evaluations'][0]
        self.assertTrue(old['stale']);self.assertEqual(old['prediction_months'],123.5)
        self.assertEqual(old['reviews'][0]['confirmed_months'],120)
        self.assertEqual(self.client.post(f"/api/evaluations/{e['id']}/reviews",json={
            'reviewer':'A','decision':'accepted','note':'不应允许审核过时结果'}).status_code,409)
        self.assertEqual(self.client.get('/api/status').json()['stats']['assessed'],0)

    def test_explanation_failure_preserves_boneage(self):
        c=self.ready();detail=self.assess_test_double(c,(110,None,'解释不可用'))
        e=detail['evaluations'][0]
        self.assertEqual(e['status'],'partial');self.assertEqual(e['prediction_months'],110)
        self.assertIsNone(e['heatmap_url'])
        self.assertEqual(e['explanation_status'],'unavailable')

    def test_inference_failure_does_not_create_prediction(self):
        c=self.ready()
        with patch.object(self.app.state.predictor,'metadata',return_value=self.model_meta()), \
             patch.object(self.app.state.predictor,'assess',side_effect=RuntimeError('验证失败')):
            self.client.post(f"/api/cases/{c['id']}/assess",json={'revision':c['revision']})
            for _ in range(100):
                e=self.client.get(f"/api/cases/{c['id']}").json()['evaluations'][0]
                if e['status']=='failed':break
                time.sleep(.01)
        self.assertEqual(e['status'],'failed');self.assertIsNone(e['prediction_months'])

    def test_cross_site_mutation_and_printable_frontend(self):
        r=self.client.post('/api/cases',headers={'Origin':'https://malicious.invalid'})
        self.assertEqual(r.status_code,403)
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertIn('骨龄研究工作台',self.client.get('/').text)
        self.assertEqual(self.client.get('/api/cases/not-a-case').status_code,404)

    def test_text_fields_cannot_inject_sql(self):
        self.ready()
        r=self.client.get('/api/cases',params={'q':"' OR 1=1 --"})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['total'],0)

    def test_unlabeled_prediction_export_is_an_artifact_not_an_accuracy(self):
        self.assertEqual(self.client.get('/api/unlabeled-predictions').status_code,404)
        path=self.settings.storage/'unlabeled-predictions.csv'
        path.write_text('Case ID,Sex,predicted_boneage_months,model_id,status\n4360,M,120,test-model,research_prediction\n')
        response=self.client.get('/api/unlabeled-predictions')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.content,path.read_bytes())
        self.assertIn('attachment',response.headers['content-disposition'])


class ImagingTest(unittest.TestCase):
    def test_letterbox_keeps_aspect_and_transform(self):
        arr,t=letterbox(Image.new('L',(100,200),128),96)
        self.assertEqual(arr.shape,(96,96))
        self.assertEqual((t['width'],t['height'],t['left'],t['top']),(48,96,24,0))
        self.assertTrue((arr[:,0]==0).all());self.assertTrue((arr[:,30]==128).all())

    def test_bad_png_and_bomb_limit(self):
        with self.assertRaises(ImageValidationError):validate_png(b'bad')
        with self.assertRaises(ImageValidationError):validate_png(png(),max_pixels=100)


if __name__=='__main__':unittest.main()
