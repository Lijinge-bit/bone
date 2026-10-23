"""Checkpoint assembly must be complete, verified, resumable and non-destructive."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import fetch_tuned_weights as fetch


class WeightFetchTests(unittest.TestCase):
    def fixture(self, root):
        parts = [b'first tensor bytes', b'second tensor bytes']
        raw = b''.join(parts)
        meta = {'model_id':'synthetic', 'checkpoint':'model.pt','checkpoint_sha256':hashlib.sha256(raw).hexdigest()}
        manifest = {'model_id':'synthetic','checkpoint_sha256':meta['checkpoint_sha256'],'bytes':len(raw),
                    'parts':[{'file':f'part{i}.bin','url':fetch.URL_PREFIX+f'part{i}.bin','bytes':len(p),
                              'sha256':hashlib.sha256(p).hexdigest()} for i,p in enumerate(parts)]}
        (root/'candidate.json').write_text(json.dumps(meta));(root/'model.parts.json').write_text(json.dumps(manifest))
        return parts,raw,manifest

    def test_reconstructs_exact_checkpoint_and_skips_existing_valid_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);parts,raw,_=self.fixture(root)
            with patch.object(fetch,'urlopen',side_effect=[io.BytesIO(p) for p in parts]) as network:
                path=fetch.fetch_weights(root)
                self.assertEqual(path.read_bytes(),raw)
                fetch.fetch_weights(root)
                self.assertEqual(network.call_count,2)

    def test_bad_part_never_publishes_model_and_verified_parts_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);parts,raw,_=self.fixture(root)
            with patch.object(fetch,'urlopen',side_effect=[io.BytesIO(parts[0]),io.BytesIO(b'bad')]):
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    fetch.fetch_weights(root)
            self.assertFalse((root/'model.pt').exists())
            with patch.object(fetch,'urlopen',return_value=io.BytesIO(parts[1])) as network:
                self.assertEqual(fetch.fetch_weights(root).read_bytes(),raw)
                self.assertEqual(network.call_count,1)

    def test_mismatch_and_untrusted_url_preserve_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,manifest=self.fixture(root);(root/'model.pt').write_bytes(b'existing')
            with patch.object(fetch,'urlopen') as network:
                with self.assertRaisesRegex(ValueError,'preserved'):
                    fetch.fetch_weights(root)
                self.assertEqual((root/'model.pt').read_bytes(),b'existing');network.assert_not_called()
                manifest['parts'][0]['url']='https://example.invalid/part'
                (root/'model.parts.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'untrusted'):
                    fetch.fetch_weights(root)
                network.assert_not_called()
                manifest['parts'][0]['url']=fetch.URL_PREFIX+manifest['parts'][0]['file']
                manifest['parts'][0]['sha256']='../' * 20 + 'aaaa'
                (root/'model.parts.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'untrusted'):
                    fetch.fetch_weights(root)
                network.assert_not_called()

    def test_local_git_parts_are_verified_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts,raw,manifest=self.fixture(root);local=root/'parts';local.mkdir()
            for p,meta in zip(parts,manifest['parts']):(local/meta['file']).write_bytes(p)
            with patch.object(fetch,'urlopen') as network:
                self.assertEqual(fetch.fetch_weights(root,local).read_bytes(),raw);network.assert_not_called()
