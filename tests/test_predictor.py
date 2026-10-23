import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from app.predictor import Predictor, ModelUnavailable

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path(os.environ.get('BONE_TEST_STORAGE',ROOT/'var'))
MODEL=STORAGE/'models'
SAMPLE=STORAGE/'images/1491b3be5411ca33dbbe5c80ec2595886b537bdceb7f77b1a3c635a1f9c586d8.png'


@unittest.skipUnless((MODEL/'active.json').is_file() and SAMPLE.is_file(),
                     'Real model/sample not prepared; run training for this integration check')
class TrainedModelIntegration(unittest.TestCase):
    def test_real_prediction_explanation_and_repeatability(self):
        predictor=Predictor(MODEL);meta=predictor.metadata()
        with Image.open(SAMPLE) as im:
            a,heat,error=predictor.assess(im.copy(),'F',meta['model_id'],meta['checkpoint_sha256'])
            b,_,_=predictor.assess(im.copy(),'F',meta['model_id'],meta['checkpoint_sha256'])
        self.assertTrue(np.isfinite(a));self.assertGreater(a,0)
        self.assertAlmostEqual(a,b,places=4)
        self.assertIsNone(error);self.assertIsNotNone(heat)
        self.assertEqual(heat.size,(1514,2044))
        self.assertGreater(np.asarray(heat)[:,:,3].max(),0)

    def test_corrupt_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            meta=json.loads((MODEL/'active.json').read_text())
            (directory/'active.json').write_text(json.dumps(meta))
            shutil.copyfile(MODEL/meta['checkpoint'],directory/meta['checkpoint'])
            with (directory/meta['checkpoint']).open('ab') as f:f.write(b'corrupt')
            with Image.open(SAMPLE) as im:
                with self.assertRaisesRegex(ModelUnavailable,'校验失败'):
                    Predictor(directory).assess(im,'F',meta['model_id'],meta['checkpoint_sha256'])

    def test_original_runtime_predictions_and_spatial_heatmaps(self):
        from scripts.check_runtime import check_samples
        reference=json.loads((ROOT/'docs/reports/runtime-reference.json').read_text(encoding='utf-8'))
        if not all((STORAGE/'images'/(r['image_sha256']+'.png')).is_file()
                   for r in reference['samples']):
            self.skipTest('Six public samples required; use the complete research bundle')
        results=check_samples(Predictor(MODEL),STORAGE,reference)
        self.assertEqual(len(results),12)


if __name__=='__main__':unittest.main()
