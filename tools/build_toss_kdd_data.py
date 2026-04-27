from pathlib import Path
import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path('/home/hun/KDD_Cup/KDD_submission/data/toss')

USER_INT_SCALAR_FIDS = [1, 3, 4] + list(range(48, 60)) + [82, 86] + list(range(92, 110))
USER_INT_ARRAY_FIDS = [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]
USER_DENSE_FIDS = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]
ITEM_INT_SCALAR_FIDS = [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
ITEM_INT_ARRAY_FIDS = [11]

SEQ_CONFIG = {
    'seq_a': {'prefix': 'domain_a_seq', 'fids': [38, 39, 40, 41, 42, 43, 44, 45, 46], 'ts_fid': 39, 'base_offset': 0},
    'seq_b': {'prefix': 'domain_b_seq', 'fids': [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 88], 'ts_fid': 67, 'base_offset': 5},
    'seq_c': {'prefix': 'domain_c_seq', 'fids': [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 47], 'ts_fid': 27, 'base_offset': 11},
    'seq_d': {'prefix': 'domain_d_seq', 'fids': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 'ts_fid': 26, 'base_offset': 17},
}

BINS = 512
ARRAY_DIM = 4
DENSE_DIM = 4
SEQ_MAX_LEN = 64


def source_path(name: str) -> Path:
    src = ROOT / f'src_{name}'
    return src if src.exists() else ROOT / name


def parse_seq(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        parts = value.split(',')
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(',')
    out = []
    for p in parts:
        if p in ('', None, 'nan', 'None'):
            continue
        try:
            out.append(abs(int(float(p))) % (BINS - 1) + 1)
        except Exception:
            continue
    return out


def bucketize_series(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(arr.min()) if len(arr) else 0.0
    max_v = float(arr.max()) if len(arr) else 0.0
    if max_v - min_v < 1e-12:
        return np.ones(len(arr), dtype=np.int64)
    scaled = np.floor((arr - min_v) / (max_v - min_v) * (BINS - 2)).astype(np.int64) + 1
    return np.clip(scaled, 1, BINS - 1)


def make_array(tokens: List[int], offset: int, length: int = ARRAY_DIM) -> List[int]:
    if not tokens:
        return []
    return [tokens[(offset + i) % len(tokens)] for i in range(min(length, len(tokens)))]


def make_dense_row(row: pd.Series, cols: List[str]) -> List[float]:
    vals: List[float] = []
    for c in cols:
        v = row[c]
        if pd.isna(v):
            vals.append(0.0)
        else:
            vals.append(float(v))
    return vals


def build_schema() -> Dict[str, object]:
    schema = {
        'user_int': [],
        'item_int': [],
        'user_dense': [],
        'seq': {},
    }
    for fid in USER_INT_SCALAR_FIDS:
        schema['user_int'].append([fid, BINS, 1])
    for fid in USER_INT_ARRAY_FIDS:
        schema['user_int'].append([fid, BINS, ARRAY_DIM])
    for fid in ITEM_INT_SCALAR_FIDS:
        schema['item_int'].append([fid, BINS, 1])
    for fid in ITEM_INT_ARRAY_FIDS:
        schema['item_int'].append([fid, BINS, ARRAY_DIM])
    for fid in USER_DENSE_FIDS:
        schema['user_dense'].append([fid, DENSE_DIM])
    for domain, cfg in SEQ_CONFIG.items():
        features = []
        for fid in cfg['fids']:
            vs = 0 if fid == cfg['ts_fid'] else BINS
            features.append([fid, vs])
        schema['seq'][domain] = {
            'prefix': cfg['prefix'],
            'ts_fid': cfg['ts_fid'],
            'features': features,
        }
    return schema


def transform_split(name: str, user_offset: int) -> int:
    src = source_path(name)
    df = pd.read_parquet(src)

    non_special = [c for c in df.columns if c not in {'clicked', 'seq'}]
    user_scalar_sources = non_special[:len(USER_INT_SCALAR_FIDS)]
    item_scalar_sources = non_special[len(USER_INT_SCALAR_FIDS):len(USER_INT_SCALAR_FIDS) + len(ITEM_INT_SCALAR_FIDS)]
    dense_source_pool = non_special[len(USER_INT_SCALAR_FIDS) + len(ITEM_INT_SCALAR_FIDS):]
    dense_groups = [dense_source_pool[i * DENSE_DIM:(i + 1) * DENSE_DIM] for i in range(len(USER_DENSE_FIDS))]
    if any(len(g) < DENSE_DIM for g in dense_groups):
        raise RuntimeError('Not enough source columns to populate dense features')

    seq_tokens = df['seq'].apply(parse_seq)

    out = pd.DataFrame(index=df.index)
    n = len(df)
    row_ids = np.arange(n, dtype=np.int64)
    day = pd.to_numeric(df.get('day_of_week', 0), errors='coerce').fillna(0).astype(np.int64).to_numpy()
    hour = pd.to_numeric(df.get('hour', 0), errors='coerce').fillna(0).astype(np.int64).to_numpy()
    base_ts = 1_700_000_000 + day * 86_400 + hour * 3_600 + row_ids

    out['user_id'] = user_offset + row_ids + 1
    item_base = pd.to_numeric(df.get('inventory_id', 0), errors='coerce').fillna(0).astype(np.int64).to_numpy()
    out['item_id'] = np.maximum(item_base, 0) + 1
    clicked = pd.to_numeric(df['clicked'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    out['label_type'] = np.where(clicked > 0.5, 2, 1).astype(np.int32)
    out['label_time'] = base_ts.astype(np.int64)
    out['timestamp'] = base_ts.astype(np.int64)

    for fid, src_col in zip(USER_INT_SCALAR_FIDS, user_scalar_sources):
        out[f'user_int_feats_{fid}'] = bucketize_series(df[src_col])

    for fid, src_col in zip(ITEM_INT_SCALAR_FIDS, item_scalar_sources):
        out[f'item_int_feats_{fid}'] = bucketize_series(df[src_col])

    for i, fid in enumerate(USER_INT_ARRAY_FIDS):
        out[f'user_int_feats_{fid}'] = [make_array(tokens, i * 3) for tokens in seq_tokens]

    out['item_int_feats_11'] = [make_array(tokens, 37) for tokens in seq_tokens]

    for fid, cols in zip(USER_DENSE_FIDS, dense_groups):
        out[f'user_dense_feats_{fid}'] = [make_dense_row(row, cols) for _, row in df[cols].iterrows()]

    for domain, cfg in SEQ_CONFIG.items():
        fids = cfg['fids']
        n_total = len(fids)
        ts_fid = cfg['ts_fid']
        base_offset = cfg['base_offset']
        for pos, fid in enumerate(fids):
            col_name = f"{cfg['prefix']}_{fid}"
            if fid == ts_fid:
                series = []
                for ts, tokens in zip(base_ts, seq_tokens):
                    values = tokens[(base_offset + pos) % n_total::n_total][:SEQ_MAX_LEN]
                    length = len(values)
                    if length == 0:
                        series.append([])
                    else:
                        start = int(ts) - 60 * length
                        series.append([start + 60 * j for j in range(length)])
                out[col_name] = series
            else:
                out[col_name] = [tokens[(base_offset + pos) % n_total::n_total][:SEQ_MAX_LEN] for tokens in seq_tokens]

    table = pa.Table.from_pandas(out, preserve_index=False)
    tmp_path = ROOT / f'kdd_{name}'
    pq.write_table(table, tmp_path)
    final_path = ROOT / name
    backup_path = ROOT / f'src_{name}'
    if not backup_path.exists():
        final_path.rename(backup_path)
    else:
        final_path.unlink(missing_ok=True)
    tmp_path.rename(final_path)
    return user_offset + n


def main() -> None:
    schema = build_schema()
    (ROOT / 'schema.json').write_text(json.dumps(schema, indent=2))
    offset = 0
    for name in ['train.parquet', 'tiny_train.parquet', 'valid.parquet', 'test.parquet']:
        offset = transform_split(name, offset)
    print('Wrote schema.json and transformed tiny_train/valid/test parquet files under', ROOT)


if __name__ == '__main__':
    main()
