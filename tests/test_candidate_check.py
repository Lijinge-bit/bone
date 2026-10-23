import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from scripts.check_candidate import public_samples, verify_heatmap


class CandidateSpatialChecks(unittest.TestCase):
    def test_original_geometry_and_spatial_alpha_are_verified(self):
        image = Image.new("L", (96, 128))
        alpha = np.tile(np.linspace(0, 160, image.width, dtype=np.uint8), (image.height, 1))
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, 0] = alpha
        rgba[:, :, 3] = alpha
        result = verify_heatmap(Image.fromarray(rgba), image, alpha)
        self.assertEqual(result["spatial_mapping_max_alpha_difference"], 0)
        self.assertGreater(result["alpha_std"], 0)
        # Same dimensions and nonempty alpha must still fail a wrong mapping.
        with self.assertRaisesRegex(RuntimeError, "valid letterbox area"):
            verify_heatmap(Image.fromarray(rgba), image, alpha[:, ::-1])
        with self.assertRaisesRegex(RuntimeError, "original image coordinates"):
            verify_heatmap(Image.fromarray(rgba).resize((128, 96)), image, alpha)

    def test_manifest_cannot_omit_public_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly the six"):
                public_samples(Path(temporary), manifest)


if __name__ == "__main__":
    unittest.main()
