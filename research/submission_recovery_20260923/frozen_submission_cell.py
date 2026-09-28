# NEW Kaggle Cell; run only after offline_bootstrap_cell.py succeeds.
# Full Kaggle Cell NOT RUN. Helpers CPU-tested using existing 14.14 artifacts;
# official conversion tested in memory. No model inference/tracker/GT execution.
# Changing this switch is an explicit decision to run both full CUDA predictions
# and the frozen tracker. This Cell does NOT upload or submit anything.
RUN_FROZEN_PIPELINE = False
assert RUN_FROZEN_PIPELINE, 'INFERENCE_AND_TRACKING_NOT_AUTHORIZED_BY_DEFAULT'
assert globals().get('BOOTSTRAP_PASS') is True

import csv
import gc
import itertools
import math
import shutil
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
import zarr
import predict_unet_transformer_mps as predictor
from biohub_cell_tracking import frozen_v9 as frozen
from tracking_cellmot.io import open_dataset, save_graph
from geffs_to_csv import geffs_to_csv
from csv_to_geffs import csv_to_geffs

WORK = Path('/kaggle/working/biohub_1415_recovery_v1')
FINAL = Path('/kaggle/working/submission.csv')
assert not WORK.exists() and not FINAL.exists(), 'Refuse to overwrite existing artifacts'
assert torch.cuda.is_available(), 'No silent CPU fallback'
for relative, r in source_manifest['files'].items():
    assert file_sha(ROOT / relative) == r['sha256'], relative
competition_roots = [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
                     Path('/kaggle/input/biohub-cell-tracking-during-development')]
test_dirs = list(dict.fromkeys((p / 'test').resolve() for p in competition_roots if (p / 'test').is_dir()))
assert len(test_dirs) == 1, ('COMPETITION_MOUNT_AMBIGUOUS_OR_MISSING', test_dirs)
TEST_DIR = test_dirs[0]
zarrs = sorted(p for p in TEST_DIR.glob('*.zarr') if p.is_dir())
# predict() uses data_dir/name and writes name.geff; open_dataset resolves stem.zarr.
# frozen_v9 uses name.geff and open_dataset(TEST_DIR/name). Keep the same stem.
# Do not mix this new run with the historical <id>.zarr.geff naming convention.
names = [p.stem for p in zarrs]
assert names and len(names) == len(set(names))
assert all(Path(n).name == n and not n.endswith('.zarr') for n in names)
shapes, scales = {}, {}
for name, p in zip(names, zarrs, strict=True):
    ds = open_dataset(TEST_DIR / name, load_image=False, require_tracks=False)
    assert Path(ds.zarr_path).resolve() == p.resolve(), 'Official IO changed dataset name resolution'
    shapes[name] = tuple(int(v) for v in ds.image_shape)
    scales[name] = tuple(float(v) for v in ds.scale)
    assert len(shapes[name]) == 4 and all(v > 0 for v in shapes[name])
    assert len(scales[name]) == 3 and all(math.isfinite(v) and v > 0 for v in scales[name])
    assert '0.001' in ds.quantiles and '0.999' in ds.quantiles, 'Frozen predictor requires stored quantiles'
WORK.mkdir()
SPLITS = WORK / 'test_splits.json'
with SPLITS.open('x') as f:
    json.dump([{'test': names}], f, indent=2)
WEIGHTS = ROOT / 'models/official_baseline/official_baseline_fold0_warm2_lr1e5/split_0/edge_predictor_best.pth'

def tree_sha(path):
    path = Path(path)
    assert not path.is_symlink()
    if path.is_file():
        return file_sha(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in path.rglob('*'))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(path).as_posix().encode() + b'\0' + bytes.fromhex(file_sha(p)) + b'\0')
    return h.hexdigest()

def validate_graph(graph, name, *, prediction=False):
    nodes, edges = graph.node_attrs(), graph.edge_attrs()
    assert {'node_id','t','z','y','x'} <= set(nodes.columns)
    points = {}
    for row in nodes.select('node_id','t','z','y','x').iter_rows(named=True):
        raw_id = row['node_id']
        node_id = int(raw_id)
        assert node_id == raw_id and node_id >= 0 and node_id not in points
        v = tuple(float(row[k]) for k in ('t','z','y','x'))
        assert all(math.isfinite(x) and 0 <= x < limit for x, limit in zip(v, shapes[name], strict=True))
        assert v[0].is_integer()
        points[node_id] = v
    incoming, outgoing, pairs = Counter(), Counter(), set()
    for row in edges.iter_rows(named=True):
        s, t = int(row['source_id']), int(row['target_id'])
        assert s == row['source_id'] and t == row['target_id']
        assert s in points and t in points and s != t and (s,t) not in pairs
        dt = points[t][0] - points[s][0]
        assert dt > 0 and (not prediction or dt == 1)
        pairs.add((s,t)); incoming[t] += 1; outgoing[s] += 1
        if prediction:
            assert math.isfinite(float(row['edge_prob'])) and 0 <= row['edge_prob'] <= 1
            assert math.isfinite(float(row['edge_dist'])) and row['edge_dist'] >= 0
    assert max(incoming.values(), default=0) <= 1
    assert max(outgoing.values(), default=0) <= 2
    return {'nodes': len(points), 'edges': len(pairs)}

def verify_directory(directory, *, prediction=False):
    assert {p.name for p in directory.glob('*.geff')} == {n + '.geff' for n in names}
    result = {}
    for name in names:
        p = directory / (name + '.geff')
        graph = frozen.load_graph(p)
        result[name] = {**validate_graph(graph, name, prediction=prediction), 'tree_sha256': tree_sha(p)}
        del graph
    return result

prediction_dirs, prediction_records, prediction_configs = {}, {}, {}
for threshold, method in [(0.995, 'warm2_det0995'), (0.9975, 'warm2_det09975')]:
    destination = WORK / 'predictions' / 'kaggle_1415' / method / 'split_0'
    assert not destination.exists(), 'predict() deletes existing GEFFs; refuse rerun'
    cfg = predictor.PredictConfig(det_threshold=threshold, pool_kernel_um=3.0, use_ilp=False)
    assert cfg.det_tta and cfg.edge_activation == 'softmax' and cfg.threshold == 0.5
    assert cfg.max_parents_per_node == 1 and cfg.max_children_per_node == 2
    prediction_configs[method] = asdict(cfg)
    previous_path, previous_user = predictor.PREDICTIONS_PATH, predictor.USERNAME
    try:
        predictor.PREDICTIONS_PATH = WORK / 'predictions'
        predictor.USERNAME = 'kaggle_1415'
        predictor.predict(data_dir=TEST_DIR, fold=0, splits_file=SPLITS, weights_path=WEIGHTS,
                          cfg=cfg, method=method, debug_video=None, unet_batch_size=4,
                          video_slice=None, evaluate=False, max_frames=None)
    finally:
        predictor.PREDICTIONS_PATH, predictor.USERNAME = previous_path, previous_user
    prediction_dirs[threshold] = destination
    prediction_records[method] = verify_directory(destination, prediction=True)
    with (WORK / (method + '_validation.json')).open('x') as f:
        json.dump(prediction_records[method], f, indent=2)
    print('CUDA_FULL_INFERENCE_AND_GEFF_PASS', method)
    gc.collect()

cfg = frozen.FROZEN_V9_CONFIG
assert cfg.det_threshold == 0.995 and cfg.high_threshold == 0.9975
assert cfg.confidence_component_limit == 9 and tuple(cfg.model_downsample_zyx) == (1.,4.,4.)
# Intentionally preserve ValueError when a global candidate class is empty.
policies = frozen.build_frozen_v9_policies(names, prediction_dirs[0.995], prediction_dirs[0.9975],
                                         train_dir=TEST_DIR, prefix_map=None, config=cfg, progress=True)
policy_counts = {field: len(getattr(policies, field)) for field in
                 ('gap_candidates','single_policy','isolated_candidates','isolated_scope','strict_two_policy')}
print('POLICY_BUILD_PASS', policy_counts)
TRACK = WORK / 'tracking_geffs'
TRACK.mkdir()
applications, track_counts = {}, {}
for name in names:
    graph, scale, application = frozen.apply_frozen_v9(name, prediction_dirs[0.995], prediction_dirs[0.9975],
                                                      policies, train_dir=TEST_DIR, config=cfg,
                                                      strict_single_policy=True)
    assert np.array_equal(np.asarray(scale), np.asarray(scales[name]))
    track_counts[name] = validate_graph(graph, name)
    out = TRACK / (name + '.geff')
    assert not out.exists()
    save_graph(graph, out, overwrite=False)
    applications[name] = asdict(application)
    del graph
track_records = verify_directory(TRACK)
assert all(track_counts[n] == {k: track_records[n][k] for k in ('nodes','edges')} for n in names)
print('FROZEN_TRACKING_GEFF_PASS')

COLUMNS = ['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
STAGING = WORK / 'submission_unverified.csv'
assert not STAGING.exists()
geffs_to_csv(TRACK, STAGING)

def semantic_digest(values):
    h = hashlib.sha256()
    for value in sorted(values):
        h.update(json.dumps(value, separators=(',',':')).encode() + b'\n')
    return h.hexdigest()

def validate_csv(path):
    # Official exporter groups rows by dataset; keep one dataset in memory.
    results, seen, row_count = {}, set(), 0
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        for name, group in itertools.groupby(reader, key=lambda r: r['dataset']):
            assert name in names and name not in seen, 'Foreign or non-contiguous dataset'
            seen.add(name)
            points, pairs = {}, []
            incoming, outgoing = Counter(), Counter()
            for row in group:
                assert None not in row and all(v is not None for v in row.values())
                values = {k: int(row[k]) for k in COLUMNS if k not in ('id','dataset','row_type')}
                # Official CSV importer ignores id entirely; no zero-origin restriction.
                row_count += 1
                if row['row_type'] == 'node':
                    nid = values['node_id']
                    assert nid >= 0 and nid not in points
                    assert values['source_id'] == values['target_id'] == -1
                    v = tuple(values[k] for k in ('t','z','y','x'))
                    # Official exporter rounds z/y/x with Polars. Do not round again
                    # or reject a valid pre-export boundary coordinate after rounding.
                    assert 0 <= v[0] < shapes[name][0]
                    points[nid] = v
                else:
                    assert row['row_type'] == 'edge'
                    assert all(values[k] == -1 for k in ('node_id','t','z','y','x'))
                    pairs.append((values['source_id'], values['target_id']))
            assert points, 'Dataset has no node rows; stop rather than fabricate'
            assert len(pairs) == len(set(pairs))
            signatures = []
            for s,t in pairs:
                assert s in points and t in points and points[s][0] < points[t][0]
                incoming[t] += 1; outgoing[s] += 1
                signatures.append((points[s], points[t]))
            assert max(incoming.values(), default=0) <= 1 and max(outgoing.values(), default=0) <= 2
            # Distinct nodes may share rounded coordinates. Digests retain multiplicity.
            # This compares coordinate/edge multisets, not graph isomorphism.
            assert len(points) == track_counts[name]['nodes'] and len(pairs) == track_counts[name]['edges']
            results[name] = {'nodes':len(points), 'edges':len(pairs),
                             'node_semantics_sha256':semantic_digest(points.values()),
                             'directed_edge_semantics_sha256':semantic_digest(signatures)}
    assert seen == set(names)
    assert row_count == sum(v['nodes'] + v['edges'] for v in results.values())
    return {'rows':row_count, 'datasets':results}

before = validate_csv(STAGING)
ROUNDTRIP = WORK / 'roundtrip_geffs'
ROUNDTRIP_CSV = WORK / 'roundtrip.csv'
assert not ROUNDTRIP.exists() and not ROUNDTRIP_CSV.exists()
csv_to_geffs(STAGING, ROUNDTRIP, overwrite=False)
assert {p.stem for p in ROUNDTRIP.glob('*.geff')} == set(names)
geffs_to_csv(ROUNDTRIP, ROUNDTRIP_CSV)
assert validate_csv(ROUNDTRIP_CSV) == before, 'Official roundtrip changed graph semantics'
for method, threshold in [('warm2_det0995',0.995),('warm2_det09975',0.9975)]:
    assert verify_directory(prediction_dirs[threshold], prediction=True) == prediction_records[method]
for relative, r in source_manifest['files'].items():
    assert file_sha(ROOT / relative) == r['sha256']
assert verify_directory(TRACK) == track_records
with STAGING.open('rb') as src, FINAL.open('xb') as dst:
    shutil.copyfileobj(src, dst)
assert file_sha(STAGING) == file_sha(FINAL)
with (WORK / 'run_manifest.json').open('x') as f:
    json.dump({'status':'CSV_AND_OFFICIAL_ROUNDTRIP_PASS', 'source_manifest_sha256':SOURCE_SHA,
               'wheel_manifest_sha256':WHEEL_MANIFEST_SHA, 'test_ids':names,
               'shapes':shapes, 'scales':scales, 'prediction_configs':prediction_configs,
               'frozen_config':asdict(cfg), 'predictions':prediction_records,
               'policy_counts':policy_counts, 'applications':applications, 'tracking':track_records,
               'csv_validation':before, 'submission_sha256':file_sha(FINAL),
               'submission_bytes':FINAL.stat().st_size, 'gt_read':False,
               'evaluator_run':False, 'competition_submission':False}, f, indent=2)
print('CSV_STRUCTURE_PASS; OFFICIAL_ROUNDTRIP_SEMANTICS_PASS')
print('SUBMISSION_FILE_READY', FINAL, FINAL.stat().st_size, file_sha(FINAL))
print('KAGGLE_VERSION_OUTPUT_VERIFIED=NO; COMPETITION_SUBMITTED=NO; GT_READ=NO')
