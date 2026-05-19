# LoopCTR Review: HyFormer / UniMixer 비교 관점

## 요약

LoopCTR은 CTR 모델의 scaling을 “파라미터 수 증가”가 아니라 “동일 layer의 반복 사용”으로 푸는 논문이다. 핵심은 recursive layer reuse를 통해 training-time compute를 늘리되, inference에서는 loop를 쓰지 않는 train-multi-loop, infer-zero-loop 전략이다.

논문의 주장은 명확하다. Transformer 기반 CTR 모델을 깊게 쌓으면 파라미터, 저장 비용, inference latency가 같이 늘어난다. LoopCTR은 같은 layer를 여러 번 반복해서 representation을 refine하고, 각 loop depth마다 supervision을 걸어 그 효과를 shared parameter에 압축한다.

우리 HyFormer 실험과 연결하면 LoopCTR은 “N-tower를 더 붙이기”보다 “현재 best backbone을 반복/공유해서 regularize하고, inference 비용은 유지하는” 방향으로 볼 수 있다. 특히 `HyFormer tref`가 best이고, WuKong/DCN/GCN 같은 외부 tower 추가가 실패한 현재 상황에서는 꽤 실험 가치가 있다.

## 논문의 문제의식

CTR/ranking 모델의 일반적인 scaling은 다음 중 하나다.

- depth 증가: layer를 더 쌓음
- width 증가: hidden dimension, embedding dimension 증가
- tower 증가: auxiliary architecture 추가
- sequence length 증가: 더 긴 user behavior 사용

하지만 산업 환경에서는 latency와 storage 제약이 강하다. parameter를 늘리는 scaling은 곧 inference cost 증가로 이어진다. LoopCTR은 이 coupling을 끊고자 한다.

```text
기존 scaling:
  more layers -> more params -> more inference cost

LoopCTR:
  same shared loop block repeated at train time
  -> more compute during training
  -> shared params absorb multi-loop benefit
  -> zero-loop inference
```

## LoopCTR 구조

### 1. Sandwich Architecture

LoopCTR은 크게 세 부분으로 나뉜다.

```text
Entry Block -> Loop Block repeated L times -> Exit Block
```

Entry Block은 heterogeneous feature projection과 grouped self-attention을 통해 feature group별 representation을 만든다. Loop Block은 shared parameter layer를 recursive하게 반복한다. Exit Block은 global token이 sequence token을 cross-attention하고 최종 click probability를 예측한다.

HyFormer와 비교하면:

```text
HyFormer:
  NS token + sequence token
  -> query generation
  -> sequence cross-attention
  -> block stack
  -> prediction

LoopCTR:
  entry representation
  -> same loop block 반복
  -> every depth prediction supervision
  -> inference에서는 loop 생략 가능
```

### 2. Prefix Attention

LoopCTR의 Loop Block은 sequential token과 global token을 비대칭적으로 섞는다.

```text
sequence token:
  sequence token끼리만 attend

global token:
  sequence + global 전체에 attend
```

이 구조는 TokenFormer의 sequence 보호와도 통한다. sequence token이 global/static token에 끌려다니지 않게 하면서, global token은 sequence 정보를 모아 prediction에 쓸 수 있다.

우리 HyFormer에서는 이것이 다음 아이디어로 연결된다.

```text
query/sequence side:
  sequence-derived query는 NS/global에 과도하게 attend하지 않음

global side:
  global/query token은 sequence를 읽어서 prediction summary를 만듦
```

이전 GCN/u-init 실패를 생각하면, label-derived graph보다 이런 attention direction 제약이 훨씬 안전하다.

### 3. Hyper-Connected Residuals

shared loop block을 반복하면 expressiveness가 부족할 수 있다. LoopCTR은 Hyper-Connected Residuals(HCR)로 각 loop iteration에서 residual flow를 input-dependent하게 조절한다.

단순 residual:

```text
h_next = h + f(h)
```

HCR 관점:

```text
h_next = adaptive_mix(previous_streams, f(adaptive_input_mix(previous_streams)))
```

핵심은 반복되는 같은 layer라도 매 loop에서 똑같이 작동하지 않게 만드는 것이다. 우리 코드에 그대로 구현하려면 부담이 있지만, 간단한 loop gate는 적용 가능하다.

```text
h_loop = h + gate_l(h) * SharedBlock(h)
```

### 4. MoE

LoopCTR은 attention projection과 FFN에 MoE를 넣어 parameter capacity를 늘린다. 다만 top-k routing과 load balancing loss가 필요하다.

우리 환경에서는 MoE는 후순위다. 이미 sparse embedding parameter가 크고 OOM이 자주 났다. 우선은 shared loop + process supervision만 확인하는 것이 맞다.

### 5. Process Supervision

LoopCTR의 핵심 학습 objective다. loop depth마다 prediction을 만들고 BCE loss를 평균낸다.

```text
L_total = 1 / (L + 1) * sum_{l=0}^L BCE(y_hat_l, y)
```

이렇게 하면 zero-loop output도 좋은 예측을 하도록 학습된다. 논문은 zero-loop inference만으로도 강한 baseline을 넘는다고 보고한다.

우리 HyFormer에 적용하면 다음과 같다.

```text
block 0 output -> aux logit_0
block 1 output -> aux logit_1
block 2 output -> main logit_2

loss = mean(BCE(logit_i, y))
```

이건 LoopCTR의 full recursive loop 없이도 바로 적용 가능한 강한 regularization이다.

## HyFormer와 비교

| 관점 | HyFormer | LoopCTR |
| --- | --- | --- |
| scaling 방식 | block/tower/sequence encoder 확장 | shared block 반복 |
| inference cost | layer/tower 추가 시 증가 | zero-loop inference로 비용 통제 |
| supervision | final logit 중심 | every loop depth process supervision |
| token 방향성 | query가 sequence에 cross-attention | global token이 sequence를 aggregate |
| sequence 보호 | 구조적으로 domain 분리 | prefix attention으로 방향 제한 |

HyFormer는 sequence domain 처리와 query generation이 강점이다. LoopCTR은 구조를 크게 키우지 않고 training-time refinement를 통해 shared parameter를 더 잘 쓰게 한다.

우리 결과에서는 다음 패턴이 있었다.

```text
HyFormer tref                     0.817137
HyFormer WuKong two tower          0.800326
HyFormer WuKong bilinear           0.799282
HyFormer LightGCN                  0.627606
HyFormer u init                    0.619317
```

즉 외부 구조를 붙이는 방향은 일반화가 좋지 않았다. LoopCTR은 외부 tower를 붙이지 않고 같은 backbone을 더 잘 훈련시키는 방향이라, 현재 실패 패턴과 다르게 접근한다.

## UniMixer / UniMax와 비교

UniMixer는 token mixing 연산을 더 유연하게 만드는 방향이고, LoopCTR은 같은 연산을 반복적으로 사용해 training compute를 늘리는 방향이다.

| 관점 | UniMixer | LoopCTR |
| --- | --- | --- |
| 핵심 질문 | token을 어떻게 섞을 것인가 | 같은 block을 반복해 더 깊게 생각하게 할 수 있는가 |
| parameter 변화 | mixer parameter 증가 | shared parameter 재사용 |
| inference 전략 | 일반적으로 변경된 mixer 그대로 사용 | train multi-loop, infer zero-loop 가능 |
| regularization | learnable mixing inductive bias | parameter sharing + process supervision |
| 우리 실험 후보 | `hyf_unimixer_*` | `hyf_loop_tref`, `hyf_process_supervision` |

둘은 경쟁 관계라기보다 조합 가능하다. 예를 들어 `tref` block 내부 mixer가 RankMixer든 UniMixer든, 같은 block을 반복 학습하고 depth-wise supervision을 거는 실험이 가능하다.

다만 먼저 확인할 것은 단순한 process supervision이다. full recursive loop는 구현 리스크가 있고 학습 시간이 늘어난다. 반면 중간 block output에 aux loss를 거는 것은 작고 안전하다.

## 우리 코드에 적용할 실험 제안

### 1순위: HyFormer tref Process Supervision

가장 현실적인 LoopCTR식 실험이다. 현재 `num_hyformer_blocks=2`라면 block별 intermediate representation에서 logit을 만든다.

```text
after query generation or block 0:
  logit_0

after block 1:
  logit_1

final:
  logit_final

loss = w0 * BCE(logit_0, y)
     + w1 * BCE(logit_1, y)
     + wf * BCE(logit_final, y)
```

권장 weight:

```text
w0 = 0.2
w1 = 0.3
wf = 1.0
```

또는 LoopCTR처럼 단순 평균도 가능하지만, 우리처럼 final score가 submission에 직접 쓰이는 경우 final weight를 더 크게 두는 편이 안전하다.

기대 효과:

- deep block이 overfit하기 전 shallow representation도 label-aligned
- final head에만 gradient가 몰리지 않음
- 추가 inference 비용 없음

### 2순위: Shared HyFormer Block Loop

현재 HyFormer block stack을 독립 block 2개로 두는 대신, 1개 shared block을 2~3회 반복한다.

```text
shared_block = MultiSeqHyFormerBlock(...)

for l in range(num_loops):
    q, ns, seq = shared_block(q, ns, seq)
    aux_logit_l = head(q, ns, seq)
```

실험 설정:

```text
train loops = 2 or 3
infer loops = 0, 1, 2 비교
```

단, LoopCTR의 핵심인 infer-zero-loop를 그대로 적용하려면 entry output만으로도 예측 가능해야 한다. HyFormer에서는 query generation 직후 logit을 만드는 aux head가 필요하다.

### 3순위: Train Multi-loop, Infer Fewer-loop

학습은 3-loop로 하고, inference/submission은 1-loop 또는 2-loop만 사용하는 방식이다.

```text
train:
  loops = 3
  loss at all depths

infer:
  loops = 1 or 2
```

우리 대회 환경에서는 inference latency보다 AUC가 더 중요할 수 있으므로, zero-loop만 고집할 필요는 없다. 하지만 train loop > infer loop가 성능을 유지하면 regularization 효과가 있다는 뜻이다.

### 4순위: Prefix Attention Direction Constraint

LoopCTR의 prefix attention은 sequence token이 global token에 끌리지 않게 하고, global token만 sequence를 읽게 한다.

HyFormer식 적용:

```text
sequence/query update:
  q attends seq only

global update:
  global/query summary attends seq + ns

prediction:
  global summary 중심
```

이 방향은 TokenFormer의 SCP 방지와도 일관된다.

### 5순위: Lightweight Loop Gate

HCR 전체 구현 대신, 반복 block output에 gate를 둔다.

```text
delta = SharedBlock(h)
gate = sigmoid(Linear([h, delta]))
h = h + gate * delta
```

shared block 반복에서 같은 transformation이 과하게 누적되는 것을 줄일 수 있다.

## 실험 우선순위

```text
1. hyf_tref_auxloss
   - block별/process supervision만 추가
   - inference 구조 동일

2. hyf_tref_shared_loop
   - independent 2 blocks -> shared 1 block x 2 loops
   - aux loss 포함

3. hyf_tref_train3_infer2
   - train loops=3, infer loops=2
   - AUC와 runtime 비교

4. hyf_tref_prefix_global
   - global token이 sequence를 읽고 sequence/query는 global influence 제한

5. hyf_tref_loop_gate
   - HCR-lite gate 추가
```

## 주의점

LoopCTR의 장점을 그대로 얻으려면 process supervision이 반드시 필요하다. 단순히 같은 HyFormer block을 여러 번 반복하면 학습 시간만 늘고 overfitting이 커질 수 있다.

또한 우리 데이터에서는 train/test gap과 label-derived feature 문제가 반복됐다. LoopCTR식 loop는 feature leakage와 무관하지만, training-time compute를 늘리므로 train AUC만 크게 오르고 test가 떨어질 수 있다. 따라서 다음 지표를 같이 봐야 한다.

```text
train AUC
valid AUC
submission AUC
block-depth별 aux AUC
infer loop depth별 AUC
runtime per epoch
```

## 결론

LoopCTR은 지금까지의 N-tower 실패 이후에 볼 만한 방향이다. 외부 tower를 더 붙이는 대신, 현재 가장 강한 `HyFormer tref` backbone을 shared-loop/process-supervision 방식으로 더 잘 훈련시키는 접근이기 때문이다.

가장 먼저 할 실험은 full LoopCTR 구현이 아니라 `hyf_tref_auxloss`다. block별 중간 head와 aux BCE만 추가하면 되고, inference cost는 그대로다. 이게 효과가 있으면 shared loop와 train-multi-loop/infer-fewer-loop로 확장하는 것이 합리적이다.

## 참고 자료

- LoopCTR arXiv: https://arxiv.org/abs/2604.19550
- LoopCTR review: https://www.themoonlight.io/en/review/loopctr-unlocking-the-loop-scaling-power-for-click-through-rate-prediction
- Hugging Face paper page: https://huggingface.co/papers/2604.19550
