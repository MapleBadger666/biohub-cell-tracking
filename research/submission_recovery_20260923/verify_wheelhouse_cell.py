# New read-only Kaggle Cell. Not executed against a Kaggle wheelhouse.
# Set ZIP_PATH to a recovered/rebuilt ZIP; do not run when it is absent.
from pathlib import Path, PurePosixPath
from email.parser import BytesParser
import hashlib
import io
import json
import zipfile

ZIP_PATH = Path('/kaggle/working/biohub_wheelhouse_py312_v1.zip')
# Original artifact locks. A rebuild is a new artifact unless these match.
EXPECTED_ZIP_SHA = 'b25787484b0a6430caaf203aa5557ffefd4e7713c00b537accaf47c4b0a1f1bc'
EXPECTED_MANIFEST_SHA = '9e132365f3c182a34d67a2c6b89d78f0bce246bf2bec6960d2f4ef824be3aafa'

def stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

assert ZIP_PATH.is_file(), 'ARTIFACT_MISSING: recover/rebuild once; do not repeat this check'
with ZIP_PATH.open('rb') as f:
    actual_zip_sha = stream_sha(f)
assert actual_zip_sha == EXPECTED_ZIP_SHA, ('NEW_OR_CHANGED_ZIP', actual_zip_sha)
with zipfile.ZipFile(ZIP_PATH) as archive:
    infos = archive.infolist()
    assert len(infos) == len({i.filename for i in infos}), 'Duplicate ZIP entries'
    for i in infos:
        p = PurePosixPath(i.filename)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
        assert ((i.external_attr >> 16) & 0o170000) != 0o120000, 'ZIP symlink'
    files = {i.filename for i in infos if not i.is_dir()}
    print('DIRECTORY_ENTRIES', [i.filename for i in infos if i.is_dir()])
    raw_manifest = archive.read('wheelhouse_manifest.json')
    assert hashlib.sha256(raw_manifest).hexdigest() == EXPECTED_MANIFEST_SHA
    manifest = json.loads(raw_manifest)
    records = manifest['wheels']
    assert len(records) == 13 and len({r['filename'] for r in records}) == 13
    assert files == {'wheelhouse_manifest.json'} | {'wheels/' + r['filename'] for r in records}
    for r in records:
        assert PurePosixPath(r['filename']).name == r['filename']
        member = 'wheels/' + r['filename']
        assert archive.getinfo(member).file_size == r['size_bytes']
        with archive.open(member) as f:
            assert stream_sha(f) == r['sha256'], member
        with zipfile.ZipFile(io.BytesIO(archive.read(member))) as wheel:
            names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
            assert len(names) == 1
            metadata = BytesParser().parsebytes(wheel.read(names[0]))
            print('WHEEL_PASS', r['filename'], r['size_bytes'], r['sha256'])
            print('METADATA', metadata['Name'], metadata['Version'], metadata['Requires-Python'])
            for requirement in metadata.get_all('Requires-Dist', []):
                print('REQUIRES', requirement)
    assert archive.testzip() is None
print('WHEELHOUSE_ZIP_INTEGRITY_PASS')
print('OFFLINE_INSTALL_VERIFIED=NO; IMPORT_SMOKE_VERIFIED=NO; FILES_WRITTEN=0')
