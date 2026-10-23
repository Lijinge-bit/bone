"""Fetch checksum-bound model parts and atomically reconstruct model.pt.

Only this repository's public HTTPS model-part URLs are accepted. Existing
weights are preserved on any failure; this command never registers a model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filelock import FileLock

URL_PREFIX = 'https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/model-parts/'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch_weights(candidate, parts_dir=None):
    candidate = Path(candidate).resolve()
    meta = json.loads((candidate / 'candidate.json').read_text(encoding='utf-8'))
    manifest = json.loads((candidate / 'model.parts.json').read_text(encoding='utf-8'))
    checkpoint = meta.get('checkpoint')
    if (checkpoint != 'model.pt' or manifest.get('checkpoint_sha256') != meta.get('checkpoint_sha256')
            or manifest.get('model_id') != meta.get('model_id')
            or not isinstance(meta.get('checkpoint_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', meta['checkpoint_sha256'])):
        raise ValueError('Parts manifest does not match the candidate checkpoint')
    parts = manifest.get('parts')
    if not isinstance(parts, list) or not parts:
        raise ValueError('Empty model-parts manifest')
    for part in parts:
        name = part.get('file')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.bin', name)
                or part.get('url') != URL_PREFIX + name or type(part.get('bytes')) is not int or part['bytes'] <= 0
                or not isinstance(part.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', part['sha256'])):
            raise ValueError('Invalid or untrusted model part')
    if sum(p['bytes'] for p in parts) != manifest.get('bytes'):
        raise ValueError('Model part byte counts do not add up')
    destination = candidate / checkpoint
    cache = candidate / '.weight-download-cache'
    with FileLock(str(candidate / '.weight-download.lock'), timeout=60):
        if destination.exists():
            if sha256(destination) != meta['checkpoint_sha256']:
                raise ValueError('Existing checkpoint differs; preserved without overwriting')
            return destination
        cache.mkdir(exist_ok=True)
        verified = []
        for number, part in enumerate(parts, 1):
            stored = cache / part['sha256']
            local = Path(parts_dir) / part['file'] if parts_dir is not None else None
            if local is not None and local.is_file() and local.stat().st_size == part['bytes'] and sha256(local) == part['sha256']:
                verified.append(local)
                continue
            if stored.is_file() and stored.stat().st_size == part['bytes'] and sha256(stored) == part['sha256']:
                verified.append(stored)
                continue
            temporary = cache / (part['sha256'] + '.' + uuid.uuid4().hex + '.tmp')
            try:
                total, digest = 0, hashlib.sha256()
                with urlopen(part['url'], timeout=60) as response, temporary.open('wb') as output:
                    while data := response.read(1024 * 1024):
                        total += len(data)
                        if total > part['bytes']:
                            raise ValueError('Model part exceeds expected size')
                        digest.update(data); output.write(data)
                    output.flush(); os.fsync(output.fileno())
                if total != part['bytes'] or digest.hexdigest() != part['sha256']:
                    raise ValueError('Downloaded model part failed its SHA256/size check')
                os.replace(temporary, stored)
            finally:
                temporary.unlink(missing_ok=True)
            verified.append(stored)
            print(f'Verified model part {number}/{len(parts)}', flush=True)
        temporary = candidate / ('.model-' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('wb') as output:
                for path in verified:
                    with path.open('rb') as stream:
                        while data := stream.read(1024 * 1024):
                            output.write(data)
                output.flush(); os.fsync(output.fileno())
            if temporary.stat().st_size != manifest['bytes'] or sha256(temporary) != meta['checkpoint_sha256']:
                raise ValueError('Reconstructed checkpoint failed its final SHA256/size check')
            if destination.exists():
                if sha256(destination) != meta['checkpoint_sha256']:
                    raise ValueError('Another checkpoint appeared; preserved without overwriting')
            else:
                os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'candidates/tail-adapt-seed20261023/export')
    parser.add_argument('--parts-dir', type=Path, default=ROOT / 'downloads/model-parts')
    args = parser.parse_args()
    try:
        path = fetch_weights(args.candidate, args.parts_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Weights not installed: {exc}\n')
    print(json.dumps({'status': 'ready', 'checkpoint': str(path), 'sha256': sha256(path),
                      'model_registered': False}, indent=2))


if __name__ == '__main__':
    main()
