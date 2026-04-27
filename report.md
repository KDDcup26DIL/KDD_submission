# Server Data Report

기준 로그:

- `server.log`: 2026-04-27 19:30 job
- `server_v2.log`: 2026-04-27 20:16 job

`server_v2.log`에서 서버 `schema.json` full dump와 row group 통계까지 확인했다. 다만 `feature_profile:` 및 train/valid label 분포 로그는 나오지 않았다. 따라서 현재 보고서는 서버 schema와 parquet metadata 중심이다.

## 서버 데이터 위치

- data dir: `/data_ams/academic_training_data`
- schema path: `/data_ams/academic_training_data/schema.json`
- schema exists: True
- parquet naming: `part-xxxxx-e514e21a-e5a7-465a-9346-3cd81c8da444-c000.gz.parquet`
- `_SUCCESS` marker 존재
- format: `raw_parquet`

## Parquet 규모

- parquet files: 1,000
- total rows: 1,010,000
- total row groups: 1,000
- 각 parquet 파일은 row group 1개를 가진다.
- 첫 파일:
  - `part-00000-e514e21a-e5a7-465a-9346-3cd81c8da444-c000.gz.parquet`
  - rows: 1,030
  - row_groups: 1
  - columns: 120

파일 row 수 분포:

- min: 975
- p50: 1,016
- max: 1,034

Row group row 수 분포:

- count: 1,000
- min: 975
- p50: 1,015
- p90: 1,028
- p99: 1,033
- max: 1,034

현재 `valid_ratio=0.1` 기준 split:

- train row groups: 900
- valid row groups: 100
- train rows: 907,381
- valid rows: 102,619

## 컬럼 구조

총 120개 컬럼.

메타/라벨:

- `user_id`: int64
- `item_id`: int64
- `label_type`: int32
- `label_time`: int64
- `timestamp`: int64

User int:

- scalar int64: `user_int_feats_1`, `3`, `4`, `48`-`59`, `82`, `86`, `92`-`109`
- list<int64>: `user_int_feats_15`, `60`, `62`, `63`, `64`, `65`, `66`, `80`, `89`, `90`, `91`

User dense:

- list<float>: `user_dense_feats_61`, `62`, `63`, `64`, `65`, `66`, `87`, `89`, `90`, `91`

Item int:

- scalar int64: `item_int_feats_5`, `6`, `7`, `8`, `9`, `10`, `12`, `13`, `16`, `81`, `83`, `84`, `85`
- list<int64>: `item_int_feats_11`

Sequence:

- `domain_a_seq_38`-`domain_a_seq_46`: list<int64>
- `domain_b_seq_67`-`domain_b_seq_79`, `domain_b_seq_88`: list<int64>
- `domain_c_seq_27`-`domain_c_seq_37`, `domain_c_seq_47`: list<int64>
- `domain_d_seq_17`-`domain_d_seq_26`: list<int64>

## Schema 해석

`user_int` / `item_int` entry는 `[fid, vocab_size, dim]` 형식이다.

예:

```json
[89, 10, 10]
```

의미:

- fid: 89
- vocab_size: 10
- dim: 10

즉 `user_int_feats_89`는 길이 10으로 padding/clip되는 list feature이고, id 값은 embedding 관점에서 `0..9` 범위를 기대한다.

`seq` entry는 `[fid, vocab_size]` 형식이다. sequence 길이는 schema가 아니라 실행 옵션 `--seq_max_lens`로 정한다.

Timestamp fid는 vocab이 `0`이다.

## 서버 Full Schema

### user_int

```text
[1, 6, 1], [3, 1725, 1], [4, 957, 1], [15, 1167, 26]
[48, 101, 1], [49, 3, 1], [50, 4, 1], [51, 144, 1]
[52, 177, 1], [53, 548, 1], [54, 2848, 1], [55, 40, 1]
[56, 1423, 1], [57, 241, 1], [58, 3, 1], [59, 17, 1]
[60, 3, 2], [62, 11, 6], [63, 49, 19], [64, 51, 26]
[65, 425, 111], [66, 1403, 150], [80, 13, 6], [82, 24, 1]
[86, 241, 1], [89, 10, 10], [90, 10, 10], [91, 10, 10]
[92, 3, 1], [93, 38, 1], [94, 7, 1], [95, 4, 1]
[96, 4, 1], [97, 4, 1], [98, 4, 1], [99, 4, 1]
[100, 4, 1], [101, 4, 1], [102, 4, 1], [103, 4, 1]
[104, 4, 1], [105, 4, 1], [106, 4, 1], [107, 4, 1]
[108, 8, 1], [109, 8, 1]
```

요약:

- feature count: 46
- vocab min: 3
- vocab max: 2,848
- list/high-dim user_int:
  - `15`: dim 26
  - `62`: dim 6
  - `63`: dim 19
  - `64`: dim 26
  - `65`: dim 111
  - `66`: dim 150
  - `80`: dim 6
  - `89`, `90`, `91`: dim 10

### item_int

```text
[5, 309, 1], [6, 912, 1], [7, 2443, 1], [8, 2131, 1]
[9, 42, 1], [10, 300, 1], [11, 21528, 20], [12, 2443, 1]
[13, 9, 1], [16, 23700, 1], [81, 3, 1], [83, 32, 1]
[84, 234, 1], [85, 1084, 1]
```

요약:

- feature count: 14
- vocab min: 3
- vocab max: 23,700
- only list item feature: `item_int_feats_11`, vocab 21,528, dim 20

### user_dense

```text
[61, 256], [62, 6], [63, 19], [64, 26], [65, 111]
[66, 150], [87, 320], [89, 10], [90, 10], [91, 10]
```

요약:

- feature count: 10
- dense total dim: 1,057
- largest dense dims:
  - `87`: 320
  - `61`: 256
  - `66`: 150
  - `65`: 111

### seq_a

- prefix: `domain_a_seq`
- timestamp fid: 39
- features:

```text
[38, 745286], [39, 0], [40, 19], [41, 11], [42, 1005]
[43, 3342], [44, 12735], [45, 7612], [46, 18]
```

요약:

- feature count: 9
- non-timestamp max vocab: 745,286 at fid 38

### seq_b

- prefix: `domain_b_seq`
- timestamp fid: 67
- features:

```text
[67, 0], [68, 28], [69, 64710562], [70, 726], [71, 2669]
[72, 10203], [73, 6761], [74, 476333], [75, 31], [76, 132080]
[77, 166], [78, 4229], [79, 11387], [88, 199678]
```

요약:

- feature count: 14
- non-timestamp max vocab: 64,710,562 at fid 69
- high-cardinality side features:
  - fid 69: 64,710,562
  - fid 74: 476,333
  - fid 88: 199,678
  - fid 76: 132,080

### seq_c

- prefix: `domain_c_seq`
- timestamp fid: 27
- features:

```text
[27, 0], [28, 73], [29, 5764358], [30, 846], [31, 6805]
[32, 7], [33, 5], [34, 1031305], [35, 2896], [36, 977479]
[37, 9433], [47, 86335515]
```

요약:

- feature count: 12
- non-timestamp max vocab: 86,335,515 at fid 47
- high-cardinality side features:
  - fid 47: 86,335,515
  - fid 29: 5,764,358
  - fid 34: 1,031,305
  - fid 36: 977,479

### seq_d

- prefix: `domain_d_seq`
- timestamp fid: 26
- features:

```text
[17, 5], [18, 966], [19, 3300], [20, 10785], [21, 4929]
[22, 404398], [23, 606041], [24, 531], [25, 15], [26, 0]
```

요약:

- feature count: 10
- non-timestamp max vocab: 606,041 at fid 23
- high-cardinality side features:
  - fid 23: 606,041
  - fid 22: 404,398

## 로컬 train_0000.parquet와 비교

로컬 `data/train_0000.parquet`도 120개 컬럼 구조는 서버와 같다. 하지만 로컬 parquet를 실제 scan해서 만든 `data/schema.json`과 서버 schema는 일부 값이 다르다.

주요 차이:

| 항목 | 서버 schema | 로컬 scan schema |
| --- | ---: | ---: |
| user_int max vocab | 2,848 | 2,844 |
| item_int max vocab | 23,700 | 35,260 |
| user_dense total dim | 1,057 | 753 |
| `user_int_feats_15` dim | 26 | 13 |
| `user_int_feats_65` dim | 111 | 49 |
| `user_int_feats_66` dim | 150 | 66 |
| `user_dense_feats_65` dim | 111 | 49 |
| `user_dense_feats_66` dim | 150 | 66 |
| `seq_a` max vocab | 745,286 | 1,201,293 |
| `seq_b` max vocab | 64,710,562 | 143,233,600 |
| `seq_c` max vocab | 86,335,515 | 278,677,640 |
| `seq_d` max vocab | 606,041 | 674,034 |

해석:

- 로컬 `train_0000.parquet`는 서버 데이터와 컬럼 구조는 맞지만, feature value 범위나 list 길이 분포가 서버 schema와 완전히 같지는 않다.
- 서버 학습/평가에 맞는 schema는 서버 `schema.json` full dump를 기준으로 해야 한다.
- 로컬 parquet에서 `max+1`로 만든 schema는 로컬 실험용으로만 보는 것이 안전하다.

## 아직 못 얻은 정보

`server_v2.log`는 아래 지점에서 끝났다.

```text
expected split by row_group order: train_rgs=900, valid_rgs=100, train_rows=907381, valid_rows=102619
```

따라서 deep profiler의 다음 출력들은 아직 확보되지 않았다.

- train/valid `label_type` 분포
- positive ratio
- timestamp / label_time min/max
- feature별 actual min/max/mean
- feature별 unique count
- feature별 OOB count
- list feature length p50/p90/p95/p99/max
- sequence domain별 실제 길이 분포

다음 서버 실행에서 deep scan이 끝까지 돌면 위 항목들을 추가로 업데이트할 수 있다.
