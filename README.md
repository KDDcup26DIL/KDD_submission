# KDD Submission Pipeline Guide

# Experiments

## Toss Tiny (110,257 row, 279.0 MB)
| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| RF | toss tiny 10 epoch | 0.556976 | 0.104141 |
| MLP | toss tiny 10 epoch | 0.620905 | 0.100910 | 
| LSTM | toss tiny 10 epoch | 0.576957 | 0.105185 |
| DCNv2 | toss tiny 10 epoch | 0.647718 | 0.098608 | 
| WuKong | toss tiny 10 epoch | 0.675493 | 0.097939 |
| HyFormer | block 2, toss tiny 10 epoch | 0.596690 | 0.100134 |
| HyFormer | block 4, toss tiny 10 epoch | 0.600007 | 0.100018 |
| HyFormerResidual | block 4, toss tiny 10 epoch | 0.595316 | 0.100052 |
| HyFormerResidualGate | block 4, toss tiny 10 epoch | 0.598816 | 0.100003 |
| HyFormerNSgate | toss 10 epoch | 0.615006 | 0.099876 | 
| EulerNet(b64) | toss tiny 10 epoch | 0.664916 | 0.100351 |
| SCV(b64) | toss tiny 10 epoch | 0.636060 | 0.107081 |
| hySCV | toss tiny 10 epoch | 0.651397 | 0.100834 |
| FinalMLP | toss tiny 10 epoch | 0.597006 | 0.101975 |
| hyformer wukong simple fusion | 10 epoch | 0.664605 | 0.098509 |
| hyformer wukong bilinear fusion | 10 epoch | 0.658183 | 0.098024 |
| hyformer wukong DCNv2 bilinear fusion | 10 epoch | 0.661140 | 0.097766 |  
| hyformer DCNv2 bilinear fusion | 10 epoch | 0.623333 | 0.099730 |
| hyformer DCNv2 simple fusion | 10 epoch | 0.631147 | 0.098718 | 
| hyformer unimixer token | 10 epoch | 0.600537 | 0.100137 |
| hyformer unimixer block | 10 epoch | 0.593689 | 0.100272 | 
| hyformer unimixer | 10 epoch | 0.602282 | 0.100493 |
| hyformer unimixer RankUp split | 10 epoch | 0.603180 | 0.100466 | 
| hyformer unimixer nlir (TokenFormer) | 10 epoch | 0.603881 | 0.099851 |
| hyformer unimixer auxloss | 10 epoch | | |

## Frappe (84,373 row, 6.8 MB)
| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| RF | frappe 10 epoch | 0.627335 | 2.188339 |
| MLP | frappe 10 epoch | 0.575936 | 2.180140 |
| LSTM | frappe 10 epoch | 0.583377 | 1.851733 | 
| DCNv2 | frappe 10 epoch | 0.583559 | 4.146001 |
| WuKong | frappe 10 epoch | 0.604436 | 1.864935 |
| HyFormer | frappe 10 epoch | 0.571346 | 0.619358 |
| HyFormer | block 4, frappe 10 epoch | 0.587791 | 0.640485 |
| HyFormerResidual | block 4, frappe 10 epoch | 0.582860 | 0.640809 |
| HyFormerResidualGate | block 4, frappe 10 epoch | 0.582632 | 0.640809 | 
| HyFormerNSgate | frappe 10 epoch | 0.600273 | 0.794707 | 
| EulerNet(b64) | frappe 10 epoch | 0.596707 | 3.624638 |
| SCV(b64) | frappe 10 epoch | 0.567228 | 6.251460 |
| FinalMLP | frappe 10 epoch | 0.558044 | 5.592541 |

## Criteo (3,951,801 row, 141.8 MB)
| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| RF | criteo tiny 10 epoch | 0.710997 | 0.137791 |
| MLP | criteo tiny 10 epoch | 0.718471 | 0.136913 |
| LSTM | criteo tiny 10 epoch | 0.717567 | 0.137013 |
| DCNv2 | criteo tiny 10 epoch | 0.716244 | 0.137127 |
| WuKong | criteo tiny 10 epoch | 0.718297 | 0.136963 | 
| HyFormer | criteo tiny 10 epoch | 0.681548 | 0.137037 |
| HyFormerNSgate | criteo tiny 10 epoch | 0.696681 | 0.139001 |
| EulerNet(b64) | criteo tiny 10 epoch | 0.685526 | 0.139992 |
| SCV(b64) | criteo tiny 10 epoch | 0.686393 | 0.141272 |
| FinalMLP | criteo tiny 10 epoch | 0.705162 | 0.138561 |

## Avazu (5,658,206 row, 273 MB)
| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| RF | 1 epoch |  |  |
| MLP | 1 epoch |  |  | 
| LSTM | 1 epoch |
| DCNv2 | 1 epoch |  |  | 
| WuKong | 1 epoch | 0.759367 | 0.369280 |
| HyFormer | 1 epoch |  |  |
| EulerNet(b64) | 1 epoch | 0.758946 | 0.370776 |
| SCV | 1 epoch |  |  |

## Sample (1,000 row, 40.1MB)
KDD sample data.

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | 10 epoch | 0.694129 | 0.534266 |
| WuKong | 10 epoch | 0.515152 | 0.387896 |
| HyFormer | block 4, 10 epoch | 0.641098 | 0.345131 | 
| HyFormerResidualGate | block 4, 10 epoch | 0.670455 | 0.340754 |
| HyFormer two | block 2, 10 epoch | 0.585227 | 0.375583 |
| HyFormer two exact | block 2, 10 epoch | 0.859375 | 0.278253 |
| hyf_tref | 10 epoch | 0.743371 | 0.328714 |

# Submission Score

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | 1 epoch [00:13:39] | 0.790978 | - | 
| WuKong | 1 epoch [00:13:11] | 0.791163 | - |
| HyFormer | 1 epoch [01:03:51] | 0.797713 | - | 
| EulerNet(1 layer) | 1 epoch [01:04:52] | 0.714584 | - |
| SCV | 1 epoch [02:11:46] | 0.72308 | - | 
| HyFormer | 10 epoch | 0.813109 | - | 
| HyFormer WuKong simple fusion | 10 epoch | 0.800326 | - |
| HyFormer wo meanpool(seq) | 10 epoch | 0.804653 | - |
| HyFormer tref | 10 epoch | 0.817137 | - |
| HyFormer tref+NS | 10 epoch | 0.810007 | - |
| HyFormer LightGCN | 10 epoch | 0.627606 | - |
| HyFormer LightGCN v2 | 10 epoch | 0.7962 | - |
| HyFormer tref CNN | 10 epoch | 0.809216 | - |
| HyFormer tref-lite | 10 epoch | 0.812782 | - |
| HyFormer u init | 10 epoch | 0.619317 | - |
| HyFormer seq init | 10 epoch | 0.809185 | - |
| HyFormer WuKong bilinear fusion | 10 epoch | 0.799282 | - |
| HyFormer WuKong DCNv2 bilinear fusion | 10 epoch | 0.79889 | - | 

## HyFormer Ablation

![HyFormer ablation AUC](./checkpoint/hyformer_ablation_hun_gpu6_260515_150309/ablation_results/ablation_auc.svg)

Run: `hyformer_ablation_hun_gpu6_260515_150309`, 10 epochs on `toss`.

| variant | best AUC | best LogLoss | description |
| --- | ---: | ---: | --- |
| `query_seq_only` | 0.618288 | 0.096820 | Query global context uses only `MeanPool(Seq_i)`. |
| `rank_none` | 0.614427 | 0.096313 | Removes RankMixer token mixing. |
| `no_time` | 0.613943 | 0.096351 | Removes time-bucket embedding. |
| `num_queries_1` | 0.613752 | 0.096543 | Uses one query per sequence instead of two. |
| `baseline` | 0.612267 | 0.096431 | Default HyFormer ablation baseline. |
| `seq_longer` | 0.610896 | 0.096613 | Uses longer/top-k sequence encoder. |
| `rank_ffn_only` | 0.610277 | 0.096411 | Keeps per-token FFN but removes RankMixer token mixing. |
| `seq_swiglu` | 0.609848 | 0.096918 | Replaces sequence self-attention with SwiGLU encoder. |
| `query_ns_only` | 0.608567 | 0.096468 | Query global context uses only user/item NS summary. |
| `rope` | 0.608465 | 0.096520 | Adds RoPE positional encoding. |
| `query_zero_seq` | 0.604653 | 0.096487 | Keeps NS summary but zeros the sequence summary in query context. |
| `no_sparse_reinit` | 0.585525 | 0.097284 | Disables sparse embedding re-initialization. |

구성요소별 의미:

- `query_context`: sequence별 query token을 만들 때 쓰는 global summary. Baseline은 `[NS summary ; MeanPool(Seq_i)]`이고, ablation에서는 `seq_only`, `ns_only`, `ns_zero_seq`를 비교했다.
- `RankMixer`: user/item sparse feature embedding을 여러 NS token으로 나눈 뒤 token mixing을 수행하는 부분. `rank_none`, `rank_ffn_only`는 이 mixing의 필요성을 확인한다.
- `time bucket`: 현재 row timestamp와 sequence timestamp 차이를 bucket embedding으로 넣는 부분. `no_time`은 이 temporal signal을 제거한다.
- `sequence encoder`: sequence token을 처리하는 encoder. `transformer`, `longer`, `swiglu`를 비교했다.
- `num_queries`: 각 sequence domain당 생성하는 query token 개수. `num_queries_1`은 query capacity를 줄인 설정이다.
- `sparse reinit`: high-cardinality sparse embedding의 과적합을 줄이기 위한 epoch-end reinitialization. `no_sparse_reinit`은 이 regularization을 제거한다.

분석:

- 가장 좋은 결과는 `query_seq_only`로, baseline `0.612267` 대비 `+0.006022` AUC를 보였다. 이 run에서는 user/item NS summary를 query 생성에 직접 넣는 것보다, 각 domain sequence의 단순 mean summary가 query conditioning에 더 잘 맞았다.
- `query_ns_only`와 `query_zero_seq`가 baseline보다 낮은 점을 보면, sequence summary는 query 생성에 필수적이다. 반대로 NS summary는 query 생성 단계에서는 오히려 noise가 될 수 있고, NS token은 이후 HyFormer block 내부에서 처리하는 정도가 더 안정적일 가능성이 있다.
- `no_sparse_reinit`의 성능 하락이 가장 크다. Sparse embedding reinitialization은 이 데이터에서 regularization 효과가 분명하며, high-cardinality feature overfitting을 억제하는 핵심 장치로 보인다.
- `rope`, `seq_swiglu`, `seq_longer`는 baseline을 넘지 못했다. 현재 설정에서는 positional/longer attention 강화보다 단순 sequence summary와 기존 transformer encoder가 더 안정적이다.
- `rank_none`과 `no_time`은 baseline보다 약간 높게 나왔지만 차이가 작다. 단일 local split 기준이므로 submission 성능까지 확인하기 전에는 구조적 우위로 단정하기 어렵다.



# 1. Environment 설정

## 1.1 conda 환경 생성
```bash
conda env create -f environment.yml
conda activate <env_name>
```

# 2. Dataset Download
```bash
cd data
gdown --fuzzy "https://drive.google.com/file/d/1rC8pwPlPYboVD__KGkS93Om2gmBfkop6/view?usp=sharing"
unzip toss_v2.zip

gdown --fuzzy "https://drive.google.com/file/d/1aDBdkknJa2cLeGz3ItDEr5oxSTW1LlCK/view?usp=sharing"
unzip frappe.zip

gdown --fuzzy "https://drive.google.com/file/d/1TdighSNhGYLPRw6junzGJcB5aoyw8jkc/view?usp=sharing"
unzip criteo.zip

gdown --fuzzy "https://drive.google.com/file/d/1GQHmnzjoajfZHoY_bLdP51wg3QrBLWxG/view?usp=sharing"
unzip Avazu.zip
```
- gdown으로 Google Drive 파일 다운로드
- dataset은 `data/{dataset_name}/` 아래에 다음 파일명으로 배치

```
data/
 └── {dataset_name}/
      ├── train.parquet
      ├── valid.parquet
      ├── test.parquet
      └── schema.json
```

- `schema.json`은 `data/{dataset_name}/schema.json`을 우선 사용하고, 없으면 `data/schema.json`을 사용

# 3. 모델 폴더 생성
```bash
./make_dir.sh {model_name}
```

생성 구조 
```
{model_name}/
 ├── {model_name}_local/
 │    ├── model_training/
 │    └── model_evaluation/
 └── {model_name}_submission/
```

# 4. 모델 구현

- sample/ 폴더 또는 hyformer 구조 참고
- 반드시 아래 인터페이스를 만족해야 함

필수 구현 요소

- forward()
- predict()
- get_sparse_params()
- get_dense_params()
- reinit_high_cardinality_params()
- num_ns attribute

# 5. Local evaluation
`run.local.sh`는 `train.py`에 `train.parquet + valid.parquet`만 전달하고,
`test.parquet`는 학습 완료 후 `infer.py`와 metric 계산에만 사용한다.

```bash
./run.local.sh {model_name} {gpu_id} {dataset_name} \
  --num_epochs 1 \
  --batch_size 64
```

example
```bash
./run.local.sh dcnv2 6 toss --num_epochs 1 --batch_size 64
./run.local.sh hyformer 0 frappe --num_epochs 1 --batch_size 256
```

# 6. Submission 파일 생성
```bash
./run.submit.sh {model_name}
```

{model_name}_local 기준으로 submission 생성

생성 구조
```
{model_name}/{model_name}_submission/
 ├── {model_name}_training/   (7 files)
 └── {model_name}_evaluation/ (3 files)
```

# 7. Hyperparameter 설정

train.py or run.sh 수정

## 7.1 train.py 수정

{model_name}_training/train.py 의 아래 부분 수정. 
```
parser.add_argument(
    '--num_epochs',
    type=int,
    default=999,
    help='Maximum number of training epochs'
)
```

## 7.2 run.sh 수정
제출 page에서 run.sh 파라미터 수정


## note
```
1. hyf_unimix_rankup_split
2. hyf_unimix_nlir
3. hyf_unimix_auxloss
4. hyf_unimix_global_token
5. 1~4 중 best 조합
6. hyf_unimix_block_schedule
7. hyf_unimix_cross_dense
```