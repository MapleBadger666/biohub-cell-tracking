# NEW Kaggle Python 3.12/Linux x86_64 Cell, STATICALLY CHECKED ONLY.
# Run before importing numpy/torch/project modules in a fresh notebook kernel.
# No inference, tracking, solver, GT, upload, or competition submission.
ALLOW_OFFLINE_INSTALL = False  # Enable only when the user elects to restore packages.
BOOTSTRAP_PASS = False  # A failed rerun must not inherit a stale success flag.

import sys
import os
import platform
import hashlib
import json
import zipfile
import subprocess
from pathlib import Path
from email.parser import BytesParser
from importlib import metadata as md
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.tags import sys_tags
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.specifiers import SpecifierSet

assert sys.platform == 'linux' and sys.version_info[:2] == (3, 12)
assert platform.machine() == 'x86_64'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
SOURCE_SHA = '615c3550cba46a5032344549d2769201af22c0425beb9349726874669ae9c865'
# For a rebuild, replace this entire block with the VERIFIED_WHEELHOUSE_LOCK
# printed by notebook 01 after ZIP verification; persist that output with the ZIP.
# These defaults identify only the historical artifact, not an arbitrary rebuild.
WHEELHOUSE_LOCK = {
    'zip_sha256': 'b25787484b0a6430caaf203aa5557ffefd4e7713c00b537accaf47c4b0a1f1bc',
    'manifest_sha256': '9e132365f3c182a34d67a2c6b89d78f0bce246bf2bec6960d2f4ef824be3aafa',
}
WHEEL_MANIFEST_SHA = WHEELHOUSE_LOCK['manifest_sha256']
WHEEL_ZIP_SHA = WHEELHOUSE_LOCK['zip_sha256']
BASE_EXPECTED = {'torch': '2.10.0+cu128', 'numpy': '2.0.2', 'pandas': '2.3.3'}
PINNED = {'zarr':'3.1.6','numcodecs':'0.15.1','donfig':'0.8.1.post1',
          'geff':'1.3.0.1.2','geff-spec':'1.2.0','polars':'1.43.2',
          'polars-runtime-32':'1.43.2','bidict':'0.23.1','ilpy':'0.6.0',
          'pyscipopt':'6.2.1','rustworkx':'0.18.1','imagecodecs':'2026.3.6',
          'tracksdata':'0.1.0rc9.dev4+g7bfeaf845'}

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def discover(filename):
    matches = []
    for directory, dirs, files in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d not in {'train', 'test', '.git'}
                   and not d.endswith(('.zarr', '.geff'))]
        if filename in files:
            matches.append(Path(directory) / filename)
    return sorted(matches)

sources = [p for p in discover('bundle_manifest.json') if file_sha(p) == SOURCE_SHA]
assert len(sources) == 1, ('SOURCE_MANIFEST_MATCHES', sources)
ROOT = sources[0].parent
source_manifest = json.loads(sources[0].read_text())
assert len(source_manifest['files']) == 29
for relative, record in source_manifest['files'].items():
    p = ROOT / relative
    assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size == record['bytes'] and file_sha(p) == record['sha256'], relative
print('SOURCE_LOCKS_PASS', ROOT)

# Prefer the mounted extracted Dataset. ZIP-only input is verified then extracted
# into a NEW working directory; no source bundle is modified or repackaged.
manifests = [p for p in discover('wheelhouse_manifest.json') if file_sha(p) == WHEEL_MANIFEST_SHA]
assert len(manifests) <= 1, ('AMBIGUOUS_WHEELHOUSE', manifests)
if manifests:
    WHEELHOUSE = manifests[0].parent
else:
    zips = [p for p in discover('biohub_wheelhouse_py312_v1.zip') if file_sha(p) == WHEEL_ZIP_SHA]
    assert len(zips) == 1, 'WHEELHOUSE_NOT_PERSISTED_OR_WRONG_IDENTITY'
    WHEELHOUSE = Path('/kaggle/working/biohub_wheelhouse_verified_input')
    assert not WHEELHOUSE.exists(), 'Never overwrite extraction'
    with zipfile.ZipFile(zips[0]) as z:
        infos = z.infolist()
        assert len(infos) == len({i.filename for i in infos})
        raw = z.read('wheelhouse_manifest.json')
        assert hashlib.sha256(raw).hexdigest() == WHEEL_MANIFEST_SHA
        wm = json.loads(raw)
        expected = {'wheelhouse_manifest.json'} | {'wheels/' + r['filename'] for r in wm['wheels']}
        assert {i.filename for i in infos if not i.is_dir()} == expected
        for r in wm['wheels']:
            assert Path(r['filename']).name == r['filename'] and '\\' not in r['filename']
            content = z.read('wheels/' + r['filename'])
            assert len(content) == r['size_bytes'] and hashlib.sha256(content).hexdigest() == r['sha256']
        WHEELHOUSE.mkdir()
        for relative in sorted(expected):
            dest = WHEELHOUSE / relative
            dest.parent.mkdir(exist_ok=True)
            with dest.open('xb') as f:
                f.write(z.read(relative))

wm = json.loads((WHEELHOUSE / 'wheelhouse_manifest.json').read_text())
records = wm['wheels']
assert len(records) == 13 and len({r['filename'] for r in records}) == 13
assert {p.name for p in (WHEELHOUSE / 'wheels').iterdir()} == {r['filename'] for r in records}
overlay = {}
wheel_paths = []
target_tags = set(sys_tags())
for r in records:
    p = WHEELHOUSE / 'wheels' / r['filename']
    assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size == r['size_bytes'] and file_sha(p) == r['sha256']
    _, _, _, tags = parse_wheel_filename(p.name)
    assert target_tags & tags, ('INCOMPATIBLE_WHEEL_TAG', p.name)
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        assert len(names) == 1
        m = BytesParser().parsebytes(z.read(names[0]))
    name = canonicalize_name(m['Name'])
    assert name not in overlay
    assert m['Version'] == PINNED[name]
    assert not m['Requires-Python'] or SpecifierSet(m['Requires-Python']).contains(platform.python_version())
    overlay[name] = (m['Version'], m.get_all('Requires-Dist', []))
    wheel_paths.append(str(p))
assert set(overlay) == set(PINNED)
for name, version in BASE_EXPECTED.items():
    assert md.version(name) == version, ('BASE_IMAGE_DRIFT', name, md.version(name))
print('WHEELHOUSE_FILES_AND_TAGS_PASS')

# Check recursive active Requires-Dist against the simulated overlay + base.
# Extras requested by a requirement propagate to the corresponding child.
# This is independent of pip check for unrelated preinstalled applications.
def closure_check(use_overlay):
    queue = [(name, frozenset()) for name in list(PINNED) + ['torch','numpy','pandas','scipy','packaging']]
    seen, versions, failures = set(), {}, []
    environment = default_environment()
    while queue:
        name, extras = queue.pop()
        name = canonicalize_name(name)
        key = (name, extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            version, requirements = overlay[name] if use_overlay and name in overlay else (md.version(name), md.requires(name) or [])
        except md.PackageNotFoundError:
            failures.append([name, 'MISSING']); continue
        versions[name] = version
        for raw in requirements:
            try:
                req = Requirement(raw)
                active = req.marker is None or any(req.marker.evaluate({**environment, 'extra': e}) for e in ({''} | set(extras)))
                if not active:
                    continue
                child = canonicalize_name(req.name)
                child_version = overlay[child][0] if use_overlay and child in overlay else md.version(child)
                if req.url or not req.specifier.contains(child_version, prereleases=True):
                    failures.append([name, raw, child_version])
                queue.append((child, frozenset(req.extras)))
            except Exception as exc:
                failures.append([name, raw, type(exc).__name__, str(exc)])
    print('PROJECT_CLOSURE', json.dumps({'versions': versions, 'failures': failures}, sort_keys=True))
    assert not failures, 'PROJECT_CLOSURE_BLOCKED: add missing compatible wheels; do not patch versions blindly'
    return versions

closure_check(True)
assert ALLOW_OFFLINE_INSTALL, 'PREINSTALL_CHECKS_PASS; installation disabled by default'
already_loaded = [m for m in ('torch','numpy','pandas','polars','zarr','tracksdata','geff') if m in sys.modules]
assert not already_loaded, ('USE_FRESH_KERNEL_BEFORE_INSTALL', already_loaded)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--only-binary=:all:', '--find-links', str(WHEELHOUSE / 'wheels'), *wheel_paths], check=True)
for name, version in {**BASE_EXPECTED, **PINNED}.items():
    assert md.version(name) == version
versions = closure_check(False)
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
print('GLOBAL_PIP_CHECK_EXIT', check.returncode, '\n', check.stdout, check.stderr)

for rel in reversed(['external/official/scripts','external/official/src','src','scripts']):
    sys.path.insert(0, str(ROOT / rel))
import importlib
project_modules = ['tracking_cellmot.io','tracking_cellmot.metrics','tracking_cellmot.models',
                   'train_unet_transformer_mps','predict_unet_transformer_mps',
                   'biohub_cell_tracking.frozen_v9','geffs_to_csv','csv_to_geffs','evaluate','dataspec','augmentations']
for name in ['torch','numpy','pandas','scipy','zarr','numcodecs','donfig','geff','geff_spec','tracksdata',
             'polars','bidict','ilpy','pyscipopt','rustworkx','imagecodecs'] + project_modules:
    module = importlib.import_module(name)
    if name in project_modules:
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(ROOT.resolve())
        relative = origin.relative_to(ROOT.resolve()).as_posix()
        assert relative in source_manifest['files']
        assert file_sha(origin) == source_manifest['files'][relative]['sha256']
    print('IMPORT_PASS', name, getattr(module, '__file__', None))
import torch
assert torch.cuda.is_available(), 'CUDA_UNAVAILABLE_NO_CPU_FALLBACK'
print('CUDA_AVAILABLE', torch.cuda.get_device_name(0), torch.version.cuda)
report = Path('/kaggle/working/biohub_offline_environment_report.json')
with report.open('x') as f:
    json.dump({'source_manifest_sha256': SOURCE_SHA, 'wheel_manifest_sha256': WHEEL_MANIFEST_SHA, 'wheelhouse_lock': WHEELHOUSE_LOCK,
               'wheelhouse_path': str(WHEELHOUSE),
               'versions': versions, 'import_smoke': 'PASS', 'cuda_available': True,
               'pip_check_exit': check.returncode, 'pip_check_output': check.stdout,
               'inference_run': False, 'gt_read': False}, f, indent=2)
BOOTSTRAP_PASS = True
print('OFFLINE_INSTALL_PASS; PROJECT_IMPORT_SMOKE_PASS; CUDA_AVAILABILITY_PASS')
print('INFERENCE_RUN=NO; TRACKING_RUN=NO; EVALUATOR_RUN=NO')
