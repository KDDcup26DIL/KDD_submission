# KDD Submission Guide

## 목적

이 문서는 `FuxiCTR/`에서 `KDD_model_training/`와 `KDD_model_evaluation/`의 서버 실행 계약에 맞는 형태로
변환해서 제출할 때 필요한 절차와 주의점을 정리한다.


### Server data Check

```
Row Group split: 900 train (907381 rows), 100 valid (102619 rows)
...
Parquet train: 907381 rows, valid: 102619 rows, batch_size=256, buffer_batches=20
```

1 Round Train data는 1,010,000 개로 9:1 split.

## 폴더 역할

- `FuxiCTR/`
  - 로컬 실험용 코드
  - 여러 모델을 빠르게 비교하는 용도
- `KDD_model_training/`
  - 서버 학습 제출 패키지
  - 서버가 `run.sh`와 `train.py`를 실행
- `KDD_model_evaluation/`
  - 서버 평가 패키지
  - 서버가 학습 산출물로부터 모델을 재구성하고 추론

## 권장 절차

### 1. FuxiCTR에서 실험 목적을 명확히 분리

FuxiCTR에서 확인할 것은 아래와 같다.

- 어떤 피처 조합이 유효한지
- 어떤 손실함수와 batch size가 유효한지
- 어떤 정규화, hidden size, optimizer가 성능에 유리한지
- 어떤 모델 아이디어를 제출 패키지에 이식할 가치가 있는지

반대로 FuxiCTR의 파일 구조, dataloader, checkpoint 형식 자체는 제출 포맷과 다를 수 있으므로 그대로 가져가면 안 된다.


### 2. 이식 대상 결정

FuxiCTR 실험 결과를 아래 두 종류로 나눠서 판단한다.

- 하이퍼파라미터만 옮기면 되는 경우
  - 예: learning rate, batch size, dropout, loss, early stopping
- 모델 구조 자체를 옮겨야 하는 경우
  - 예: cross layer, attention block, sequence encoder 변경, embedding 처리 변경

첫 번째는 비교적 쉽고 안전하다.
두 번째는 `training`과 `evaluation`을 동시에 수정해야 하므로 리스크가 크다.


### 3. 학습 코드에 반영

수정 위치는 보통 아래다.

- `KDD_submission/KDD_model_training/model.py`
  - 모델 구조 반영
- `KDD_submission/KDD_model_training/train.py`
  - argparse, 기본 하이퍼파라미터, 모델 생성 인자 반영
- `KDD_submission/KDD_model_training/trainer.py`
  - 학습 루프, loss, metric, checkpoint 정책 반영
- `KDD_submission/KDD_model_training/run.sh`
  - 서버 기본 실행 인자 반영

FuxiCTR에서 잘 되었던 설정이라도, 제출용 코드의 입력 형식은 `PCVRParquetDataset`과 `schema.json` 계약을 따라야 한다.
데이터 입출력 계약은 가급적 건드리지 않는 편이 안전하다.


### 4. 평가 코드에 동일 구조 반영

학습 코드만 바꾸면 안 된다.

최소한 아래 파일도 학습 코드와 동기화해야 한다.

- `KDD_submission/KDD_model_evaluation/model.py`
- `KDD_submission/KDD_model_evaluation/infer.py`
- 필요 시 `KDD_submission/KDD_model_evaluation/dataset.py`

특히 `infer.py`는 `train_config.json`을 읽어 모델을 다시 만든다.
따라서 학습 시 새로 도입한 인자가 있으면:

- `train.py` argparse에 추가
- `trainer.py`가 저장하는 `train_config.json`에 포함
- `infer.py`의 fallback / resolve 로직에도 반영

이 세 군데가 맞아야 한다.


### 5. checkpoint self-contained 유지

현재 구조상 checkpoint 디렉터리에는 보통 아래 파일이 같이 있어야 한다.

- `model.pt`
- `schema.json`
- `train_config.json`
- 선택적 `ns_groups.json`

이 파일들이 맞아야 평가 서버가 모델을 재구성할 수 있다.

FuxiCTR의 checkpoint 포맷을 그대로 복사해 쓰는 방식은 추천하지 않는다.
대회 포맷은 `infer.py`가 기대하는 self-contained checkpoint를 기준으로 맞추는 편이 안전하다.


## 실제 변환 패턴

### 패턴 A: 하이퍼파라미터만 참고

가장 안전한 방식이다.

예:

- FuxiCTR에서 batch size 128, lr 1e-3, dropout 0.1이 좋았음
- 이를 `KDD_model_training/train.py` 기본값이나 `run.sh` 실행 인자로 반영

장점:

- 평가 코드 수정 범위가 작다
- shape mismatch 위험이 낮다


### 패턴 B: 구조 아이디어만 가져오기

예:

- FuxiCTR의 DCNv2 아이디어를 참고해 cross feature block 추가
- FiBiNET의 bilinear interaction 아이디어 일부 이식

이 경우 FuxiCTR 코드를 그대로 복붙하는 것보다,
`PCVRHyFormer` 입력 구조에 맞게 다시 구현하는 편이 안전하다.

이유:

- 입력 타입이 다름
- sequence 처리 구조가 다름
- checkpoint 로딩 계약이 다름
- evaluation 코드도 같은 구조를 알아야 함


### 패턴 C: FuxiCTR 모델을 거의 그대로 옮기기

가장 위험하다.

문제:

- FuxiCTR dataloader 계약과 현재 parquet schema 계약이 다름
- training/evaluation 양쪽에 같은 구조를 복제해야 함
- inference 출력 형식도 따로 맞춰야 함
- 실전 제출 직전에는 디버깅 비용이 커짐

특별한 이유가 없으면 피하는 편이 좋다.


## 가장 중요한 주의점

### 1. training과 evaluation의 모델 구조는 반드시 같아야 함

이 구조는 서버 평가 시 `infer.py`가 다시 모델을 만들고 `model.pt`를 strict load 한다.

따라서 아래 중 하나라도 어긋나면 실패할 수 있다.

- layer 수
- hidden size
- embedding dimension
- tokenizer 방식
- seq encoder 타입
- feature grouping 방식


### 2. `train_config.json`이 사실상 구조 계약이다

학습 때 사용한 하이퍼파라미터가 `train_config.json`에 남고,
평가 시 `infer.py`가 이것을 기준으로 모델을 재생성한다.

새 인자를 추가했는데 `infer.py`가 모르면:

- fallback default가 적용되거나
- shape mismatch가 나거나
- strict load가 실패할 수 있다


### 3. 데이터 포맷 계약은 함부로 바꾸지 말 것

`dataset.py`는 parquet + `schema.json` 기반 입력을 가정한다.

아래는 가능하면 유지하는 편이 안전하다.

- parquet 컬럼명 규칙
- `schema.json` 구조
- sequence domain 이름
- label / user_id 처리 방식

서버도 같은 계약을 기대할 가능성이 높다.


### 4. 예측 출력 형식을 반드시 맞출 것

평가 결과는 `predictions.json`으로 저장되어야 한다.

현재 `infer.py` 기준 형식:

```json
{
  "predictions": {
    "user_id_1": 0.123,
    "user_id_2": 0.456
  }
}
```

모델이 아무리 좋아도 출력 형식이 다르면 제출 실패 가능성이 있다.


### 5. checkpoint 경로와 sidecar 파일 저장 로직을 유지할 것

`trainer.py`는 best checkpoint 저장과 함께 sidecar 파일을 복사한다.

이 로직을 깨면 evaluation 단계에서:

- `schema.json` 없음
- `train_config.json` 없음
- `ns_groups.json` 없음

같은 문제가 생길 수 있다.


### 6. FuxiCTR 성능과 제출 성능은 바로 같지 않다

이유:

- 데이터 split 방식이 다를 수 있음
- metric 계산 시점이 다를 수 있음
- sequence 처리 구현이 다를 수 있음
- 모델 구조를 완전히 동일하게 못 옮길 수 있음

따라서 FuxiCTR는 탐색 도구로 보고, 최종 성능 검증은 제출 패키지 기준으로 다시 해야 한다.


## 추천 작업 순서

1. FuxiCTR에서 설정/아이디어를 정리한다.
2. 그중 제출 패키지에 안전하게 옮길 수 있는 것만 고른다.
3. `KDD_model_training`에 먼저 반영한다.
4. 같은 구조를 `KDD_model_evaluation`에도 반영한다.
5. `train_config.json` 기반 복원이 되도록 `infer.py`를 점검한다.
6. 로컬에서 학습 후 checkpoint 디렉터리를 기준으로 추론까지 직접 검증한다.
7. `predictions.json` 형식이 맞는지 확인한 뒤 제출한다.


## 제출 전 체크리스트

- `run.sh`가 서버 환경에서 바로 실행 가능한가
- `train.py`가 env var 기반 경로를 제대로 받는가
- 학습 후 `model.pt`가 생성되는가
- checkpoint 디렉터리에 `schema.json`이 들어가는가
- checkpoint 디렉터리에 `train_config.json`이 들어가는가
- `ns_groups.json`이 필요하면 같이 저장되는가
- `infer.py`가 해당 checkpoint를 strict load 할 수 있는가
- 평가 데이터로 `predictions.json`이 생성되는가
- `predictions.json` key/value 형식이 서버 계약과 맞는가
