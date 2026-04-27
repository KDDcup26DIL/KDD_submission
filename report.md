# Server Log Report

## 서버 데이터셋 구조

데이터 디렉터리:

- `/data_ams/academic_training_data`
- 존재 여부: True
- schema 경로: `/data_ams/academic_training_data/schema.json`
- schema 존재 여부: True

디렉터리 첫 entry:

- `_SUCCESS`
- `part-00000-e514e21a-e5a7-465a-9346-3cd81c8da444-c000.gz.parquet`
- `part-00001-e514e21a-e5a7-465a-9346-3cd81c8da444-c000.gz.parquet`
- 이후 `part-xxxxx-...gz.parquet` 형식

Parquet 메타 정보:

- parquet file count: 1000
- total rows: 1,010,000
- total row groups: 1000
- 첫 parquet 파일:
  - name: `part-00000-e514e21a-e5a7-465a-9346-3cd81c8da444-c000.gz.parquet`
  - rows: 1030
  - row_groups: 1
  - columns: 120

학습/검증 split:

- row group 기준 split
- train row groups: 900
- valid row groups: 100
- train rows: 907,381
- valid rows: 102,619

## Schema 요약

- `user_int`: 46 features
- `item_int`: 14 features
- `user_dense`: 10 features
- sequence domains: `seq_a`, `seq_b`, `seq_c`, `seq_d`

Vocab summary:

- user_int vocab:
  - count: 46
  - min: 3
  - max: 2,848
  - unique: 26
- item_int vocab:
  - count: 14
  - min: 3
  - max: 23,700
  - unique: 13

Sequence vocab summary:

- `seq_a`
  - prefix: `domain_a_seq`
  - timestamp fid: 39
  - features: 9
  - vocab count: 9
  - min vocab: 0
  - max vocab: 745,286
  - unique vocab sizes: 9
- `seq_b`
  - prefix: `domain_b_seq`
  - timestamp fid: 67
  - features: 14
  - vocab count: 14
  - min vocab: 0
  - max vocab: 64,710,562
  - unique vocab sizes: 14
- `seq_c`
  - prefix: `domain_c_seq`
  - timestamp fid: 27
  - features: 12
  - vocab count: 12
  - min vocab: 0
  - max vocab: 86,335,515
  - unique vocab sizes: 12
- `seq_d`
  - prefix: `domain_d_seq`
  - timestamp fid: 26
  - features: 10
  - vocab count: 10
  - min vocab: 0
  - max vocab: 606,041
  - unique vocab sizes: 10

## 컬럼 구조

총 120개 컬럼.

메타/라벨 컬럼:

- `user_id`: int64
- `item_id`: int64
- `label_type`: int32
- `label_time`: int64
- `timestamp`: int64

User int columns:

- scalar int64: `user_int_feats_1`, `3`, `4`, `48`-`59`, `82`, `86`, `92`-`109`
- list<int64>: `user_int_feats_15`, `60`, `62`, `63`, `64`, `65`, `66`, `80`, `89`, `90`, `91`

User dense columns:

- list<float>: `user_dense_feats_61`, `62`, `63`, `64`, `65`, `66`, `87`, `89`, `90`, `91`

Item int columns:

- scalar int64: `item_int_feats_5`, `6`, `7`, `8`, `9`, `10`, `12`, `13`, `16`, `81`, `83`, `84`, `85`
- list<int64>: `item_int_feats_11`

Sequence columns:

- `domain_a_seq_38`-`domain_a_seq_46`: list<int64>
- `domain_b_seq_67`-`domain_b_seq_79`, `domain_b_seq_88`: list<int64>
- `domain_c_seq_27`-`domain_c_seq_37`, `domain_c_seq_47`: list<int64>
- `domain_d_seq_17`-`domain_d_seq_26`: list<int64>