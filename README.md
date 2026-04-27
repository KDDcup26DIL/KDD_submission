# KDD Submission Pipeline Guide

# Experiments

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | toss tiny 10 epoch [00:06:16] | 0.625425 | 0.100677 | 
| HyFormer | toss tiny 10 epoch [00:12:53] | 0.602661 | 0.100119 | 
| WuKong | toss tiny 10 epoch [00:03:07] | 0.634248 | 0.099390 |

# Submission Score

| model | settings | AUC | LogLoss |
| --- | --- | --- | --- |
| DCNv2 | 1 epoch [00:13:39] | 0.790978 | - | 
| HyFormer | 1 epoch [01:03:51] | 0.797713 | - | 
| WuKong | 1 epoch [00:13:11] | 0.791163 | - |

# 1. Environment 설정

## 1.1 conda 환경 생성
```bash
conda env create -f environment.yml
conda activate <env_name>
```

# 2. Dataset Download
```bash
cd data
gdown --fuzzy "https://drive.google.com/file/d/1p6H-987ZFKEWue-Vpi9-PUwNd8L75STR/view?usp=sharing"
unzip toss.zip
```
- gdown으로 Google Drive 파일 다운로드
- toss.zip 압축 해제 후 parquet 데이터 사용

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
```bash
./run.local.sh {model_name} {gpu_id} \
  --num_epochs 1 \
  --batch_size 64
```

example
```bash
./run.local.sh dcnv2 6 --num_epochs 1 --batch_size 64
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
