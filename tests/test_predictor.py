import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from app.predictor import Predictor, ModelUnavailable

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'var/models'
SAMPLE=Path('/workspace/rsna-image-review/train1377.png')
if not SAMPLE.exists():
    SAMPLE=ROOT/'var/images/1491b3be5411ca33dbbe5c80ec2595886b537bdceb7f77b1a3c635a1f9c586d8.png'


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


if __name__=='__main__':unittest.main()
