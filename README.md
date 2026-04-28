# KDD Submission Pipeline Guide

# Experiments

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | toss tiny 10 epoch | 0.647718 | 0.098608 | 
| WuKong | toss tiny 10 epoch | 0.675493 | 0.097939 |
| HyFormer | toss tiny 10 epoch | 0.596690 | 0.100134 | 
| DCNv2 | frappe 10 epoch | 0.583559 | 4.146001 |
| WuKong | frappe 10 epoch | 0.604436 | 1.864935 |
| HyFormer | frappe 10 epoch | 0.571346 | 0.619358 |


# Submission Score

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | 1 epoch [00:13:39] | 0.790978 | - | 
| WuKong | 1 epoch [00:13:11] | 0.791163 | - |
| HyFormer | 1 epoch [01:03:51] | 0.797713 | - | 


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
