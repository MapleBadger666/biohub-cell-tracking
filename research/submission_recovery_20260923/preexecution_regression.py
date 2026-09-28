"""Read-only CPU compatibility regression. No inference/tracker/evaluator calls.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <this file>
All conversion is in memory; no preexisting artifacts are written.
"""
import ast
import csv
import hashlib
import inspect
import io
import itertools
import json
import math
import sys
import zipfile
import contextlib
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
PROJECT = D.parents[1]
ROOT = PROJECT / 'scratchpad/notebook14_14_15a_kaggle_bundle_v1/bundle'

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest_path = ROOT / 'bundle_manifest.json'
assert file_sha(manifest_path) == '615c3550cba46a5032344549d2769201af22c0425beb9349726874669ae9c865'
source_manifest = json.loads(manifest_path.read_text())
def check_source():
    assert len(source_manifest['files']) == 29
    for relative, record in source_manifest['files'].items():
        p = ROOT / relative
        assert p.stat().st_size == record['bytes'] and file_sha(p) == record['sha256']
check_source()
for rel in ['external/official/scripts', 'external/official/src', 'src', 'scripts']:
    sys.path.insert(0, str(ROOT / rel))
import polars as pl
import pandas as pd
import predict_unet_transformer_mps as predictor
from biohub_cell_tracking import frozen_v9 as frozen
from tracking_cellmot.io import open_dataset, save_graph
from geffs_to_csv import graph_to_rows
from csv_to_geffs import build_graph_from_rows

for cls in [predictor.PredictConfig, frozen.FrozenV9Policies, frozen.FrozenV9Application]:
    assert is_dataclass(cls)
cfg = predictor.PredictConfig(det_threshold=.995, pool_kernel_um=3., use_ilp=False)
assert (cfg.max_parents_per_node, cfg.max_children_per_node) == (1, 2)
json.dumps(asdict(cfg))
policies = frozen.FrozenV9Policies(*(pd.DataFrame() for _ in range(5)))
assert all(isinstance(getattr(policies, f), pd.DataFrame) for f in
           ['gap_candidates','single_policy','isolated_candidates','isolated_scope','strict_two_policy'])
app = frozen.FrozenV9Application(**{f.name: ('probe' if f.name == 'dataset' else 0)
                                      for f in fields(frozen.FrozenV9Application)})
json.dumps(asdict(app))
assert inspect.signature(save_graph).parameters['overwrite'].default is True
print('PASS interfaces: Polars graph tables, dataclasses, official converters/save signature')

# Load only helper functions, never the executable pipeline top level.
tree = ast.parse((D / 'frozen_submission_cell.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
exec(compile(ast.Module(body=functions, type_ignores=[]), '<reviewed-helpers>', 'exec'), globals())
COLUMNS = ['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
TEST = PROJECT / 'data/raw/competition/biohub-cell-tracking-during-development/test'
zarrs = sorted(TEST.glob('*.zarr'))
names = [p.stem for p in zarrs]
assert len(names) == 4
shapes = {}
for p in zarrs:
    a = open_dataset(TEST / p.stem, load_image=False, require_tracks=False)
    b = open_dataset(p, load_image=False, require_tracks=False)
    assert a.zarr_path == b.zarr_path == p
    assert a.image is None and a.tracks is None and b.tracks is None
    assert a.image_shape == b.image_shape
    shapes[p.stem] = a.image_shape
print('PASS actual test metadata: stem and .zarr resolve to identical Zarr; no image/GT load')

# Execute the unchanged orchestration with inference/model/save spies only.
# Patch mkdir to prevent writing anything; path joins remain real pathlib.
for suffix in ['', '.zarr']:
    recorded = []
    def video_spy(model, path, device, **kwargs):
        recorded.append(('input', path))
        assert kwargs['max_frames'] is None
        return [], []
    def save_spy(graph, path):
        recorded.append(('output', path))
    with patch.object(predictor, 'PREDICTIONS_PATH', D / '__no_write_probe__'), \
         patch.object(predictor, 'USERNAME', 'probe'), \
         patch.object(predictor, 'load_model', return_value=(None, 1, (1,4,4))), \
         patch.object(predictor, 'predict_video', side_effect=video_spy), \
         patch.object(predictor, 'build_graph', return_value=object()), \
         patch.object(predictor, 'save_graph', side_effect=save_spy), \
         patch.object(predictor, 'evaluate_run', side_effect=AssertionError('GT prohibited')), \
         patch.object(predictor.torch.cuda, 'is_available', return_value=False), \
         patch.object(predictor.torch.backends.mps, 'is_available', return_value=False), \
         patch.object(Path, 'mkdir'):
        predictor.predict(TEST, 0, SimpleNamespace(read_text=lambda: json.dumps([{'test':[names[0]+suffix]}])),
                          Path('UNUSED_WEIGHTS'), cfg, method='probe', evaluate=False, max_frames=None)
    assert recorded[0] == ('input', TEST / (names[0]+suffix))
    assert recorded[1][1].name == names[0]+suffix+'.geff'
assert not (D / '__no_write_probe__').exists()
print('PASS real predict orchestration with spies only: literal names, output naming; NO inference')

TRACK = PROJECT / 'scratchpad/notebook14_14_14_submission_v1/tracking_geffs'
track_counts = {}
artifact_locks = {}
for directory, prediction in [
    (PROJECT / 'predictions/heqiuyan/submission_v1_warm2_det0995/split_0', True),
    (PROJECT / 'predictions/heqiuyan/submission_v1_warm2_det09975/split_0', True),
    (TRACK, False),
]:
    for name in names:
        path = directory / (name+'.geff')
        artifact_locks[path] = tree_sha(path)
        graph = frozen.load_graph(path)
        assert isinstance(graph.node_attrs(), pl.DataFrame) and isinstance(graph.edge_attrs(), pl.DataFrame)
        counts = validate_graph(graph, name, prediction=prediction)
        if not prediction:
            track_counts[name] = counts
        print('PASS graph', directory.name, name, counts)

class MemoryCSV:
    def __init__(self, text): self.text = text
    def open(self, **kwargs): return io.StringIO(self.text)

def validate_frame(frame):
    # Official exporter prepends id; importer ignores it. Arbitrary origin is legal.
    return validate_csv(MemoryCSV(frame.with_row_index('id', offset=17).write_csv()))

all_original, all_rebuilt = [], []
for name in names:
    original = graph_to_rows(frozen.load_graph(TRACK / (name+'.geff')), name)
    rebuilt = build_graph_from_rows(original.filter(pl.col('row_type')=='node'),
                                    original.filter(pl.col('row_type')=='edge'))
    restored = graph_to_rows(rebuilt, name)
    all_original.append(original); all_rebuilt.append(restored)
assert validate_frame(pl.concat(all_original)) == validate_frame(pl.concat(all_rebuilt))
historical_csv = PROJECT / 'submissions/submission_v1_frozen_v9_warm2.csv'
historical_roundtrip = PROJECT / 'scratchpad/notebook14_14_14_submission_v1/14_14d_roundtrip.csv'
artifact_locks[historical_csv] = file_sha(historical_csv)
artifact_locks[historical_roundtrip] = file_sha(historical_roundtrip)
before = validate_csv(historical_csv)
assert validate_csv(historical_roundtrip) == before == validate_frame(pl.concat(all_original))
print('PASS official in-memory GEFF→rows→graph→rows + existing disk roundtrip:', before['rows'], 'rows')

# Isolated duplicate-coordinate and official fractional rounding regression.
saved_names, saved_shapes, saved_counts = names, shapes, track_counts
names, shapes, track_counts = ['probe'], {'probe':(3,10,10,10)}, {'probe':{'nodes':3,'edges':1}}
nodes = pl.DataFrame({'node_id':[8,19,42], 't':[0,0,1], 'z':[9.6,9.6,1.5],
                      'y':[2.5,2.5,3.5], 'x':[1.5,1.5,2.5]})
edges = pl.DataFrame({'source_id':[19], 'target_id':[42]})
g = build_graph_from_rows(nodes, edges)
rows = graph_to_rows(g, 'probe')
assert rows.filter(pl.col('row_type')=='node')['z'].to_list() == [10,10,2]
one = validate_frame(rows)
g2 = build_graph_from_rows(rows.filter(pl.col('row_type')=='node'), rows.filter(pl.col('row_type')=='edge'))
assert validate_frame(graph_to_rows(g2, 'probe')) == one
bad = rows.with_columns(pl.when(pl.col('row_type')=='edge').then(999).otherwise(pl.col('target_id')).alias('target_id'))
try:
    validate_frame(bad)
except AssertionError:
    pass
else:
    raise AssertionError('Dangling edge was accepted')
names, shapes, track_counts = saved_names, saved_shapes, saved_counts
print('PASS CSV: duplicate coordinates, id origin 17, official rounding; dangling endpoint rejected')

# Test the actual recursive discovery function with realistic mounted wrappers.
bootstrap = ast.parse((D/'offline_bootstrap_cell.py').read_text())
discover_ast = next(n for n in bootstrap.body if isinstance(n,ast.FunctionDef) and n.name=='discover')
import os
exec(compile(ast.Module(body=[discover_ast],type_ignores=[]), '<actual-discovery>', 'exec'), globals())
walk = [('/kaggle/input/datasets/user/private/v1', ['nested'], []),
        ('/kaggle/input/datasets/user/private/v1/nested', ['wheels'], ['wheelhouse_manifest.json']),
        ('/kaggle/input/private-zip', [], ['biohub_wheelhouse_py312_v1.zip'])]
with patch.object(os,'walk',return_value=walk):
    assert discover('wheelhouse_manifest.json') == [Path(walk[1][0])/'wheelhouse_manifest.json']
    assert discover('biohub_wheelhouse_py312_v1.zip') == [Path(walk[2][0])/'biohub_wheelhouse_py312_v1.zip']
print('PASS discovery fixtures: nested Dataset and ZIP-only mounts (not remote evidence)')

# Exercise the actual ZIP verifier entirely in memory, including explicit directory
# entries. These synthetic files are not a wheelhouse and cannot be installed.
wheel_bytes, records = {}, []
for index in range(13):
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as archive:
        archive.writestr(f'probe{index}-1.0.dist-info/METADATA',
                         f'Name: probe{index}\nVersion: 1.0\nRequires-Python: >=3.12\n')
    filename = f'probe{index}-1.0-py3-none-any.whl'
    wheel_bytes[filename] = content.getvalue()
    records.append({'filename': filename, 'size_bytes': len(content.getvalue()),
                    'sha256': hashlib.sha256(content.getvalue()).hexdigest()})
raw_manifest = json.dumps({'wheels':records}).encode()
def fixture_zip(corrupt=False):
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w') as archive:
        archive.writestr('wheels/', b'')
        archive.writestr('wheelhouse_manifest.json', raw_manifest)
        for i,(name,body) in enumerate(wheel_bytes.items()):
            archive.writestr('wheels/'+name, body + (b'changed' if corrupt and i==0 else b''))
    return result.getvalue()
verifier = (D/'verify_wheelhouse_cell.py').read_text()
verifier_tree = ast.parse(verifier)
omit = {'ZIP_PATH','EXPECTED_ZIP_SHA','EXPECTED_MANIFEST_SHA'}
verifier_tree.body = [n for n in verifier_tree.body if not (
    isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in omit for t in n.targets))]
for corrupt in [False, True]:
    payload = fixture_zip(corrupt)
    fake_path = SimpleNamespace(is_file=lambda:True, open=lambda *a,**k:io.BytesIO(payload))
    actual_zipfile = zipfile.ZipFile
    def virtual_zip(path,*args,**kwargs):
        return actual_zipfile(io.BytesIO(payload) if path is fake_path else path,*args,**kwargs)
    namespace = {'ZIP_PATH':fake_path, 'EXPECTED_ZIP_SHA':hashlib.sha256(payload).hexdigest(),
                 'EXPECTED_MANIFEST_SHA':hashlib.sha256(raw_manifest).hexdigest()}
    failed = False
    try:
        with patch.object(zipfile,'ZipFile',side_effect=virtual_zip), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(verifier_tree,'<actual-ZIP-verifier>','exec'),namespace)
    except AssertionError:
        failed = True
    assert failed == corrupt
builder=json.loads((D/'01_wheelhouse_rebuild_NOT_RUN.ipynb').read_text())
assert 'REBUILT_MANIFEST_SHA = sha256_file(manifest_path)' in ''.join(builder['cells'][1]['source'])
assert 'EXPECTED_MANIFEST_SHA = REBUILT_MANIFEST_SHA' in ''.join(builder['cells'][2]['source'])
assert "'manifest_sha256': EXPECTED_MANIFEST_SHA" in ''.join(builder['cells'][2]['source'])
print('PASS actual ZIP verifier: directory entries accepted; corrupted wheel rejected; rebuild lock handoff syntax')

for p in D.glob('*.py'): compile(p.read_text(), str(p), 'exec')
for p in D.glob('*NOT_RUN.ipynb'):
    nb=json.loads(p.read_text())
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code': compile(''.join(cell['source']),f'{p.name}:cell{i}','exec')
nb=json.loads((D/'02_offline_submission_NOT_RUN.ipynb').read_text())
assert ''.join(nb['cells'][1]['source']) == (D/'offline_bootstrap_cell.py').read_text()
assert ''.join(nb['cells'][2]['source']) == (D/'frozen_submission_cell.py').read_text()
check_source()
for p, digest in artifact_locks.items(): assert tree_sha(p) == digest
print('PASS syntax, notebook/helper sync, all 29 source hashes, 12 input GEFF and 2 CSV hashes unchanged')
print('KAGGLE_OFFLINE_INSTALL=NOT_VERIFIED; CUDA_INFERENCE=NOT_RUN; FROZEN_TRACKER=NOT_RUN; GT=NOT_READ')
