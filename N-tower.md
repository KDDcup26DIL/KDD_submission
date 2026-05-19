# N-tower Experiment Plan

## Current Status

지금까지 submission 결과를 보면 단일 HyFormer 계열에서는 `HyFormer tref`가 가장 좋다.

| model | setting | submission AUC | note |
| --- | --- | ---: | --- |
| HyFormer | 10 epoch | 0.813109 | current base |
| HyFormer tref | 10 epoch | 0.817137 | current best single model |
| HyFormer tref-lite | 10 epoch | 0.812782 | refiner 제거, gain 소실 |
| HyFormer WuKong two tower | 10 epoch | 0.800326 | simple fusion two-tower 실패 |
| HyFormer tref CNN | 10 epoch | 0.809216 | sequence CNN summary 실패 |
| HyFormer seq init | 10 epoch | 0.809185 | sequence init prior는 도움 안 됨 |
| HyFormer u init | 10 epoch | 0.619317 | click/user-history init 실패 |
| HyFormer LightGCN | 10 epoch | 0.627606 | label/click graph mismatch로 실패 |
| HyFormer LightGCN v2 | 10 epoch | 0.796200 | label-free graph도 baseline 미만 |

핵심 해석:

- `tref`의 이득은 크지 않지만 가장 안정적이다.
- `wo meanpool(seq)`가 0.804653으로 떨어졌기 때문에 sequence mean summary는 유지해야 한다.
- WuKong, CNN, GCN, user-init처럼 강한 보조 모듈을 직접 붙인 실험은 대부분 baseline보다 나빴다.
- 따라서 N-tower는 main tower를 대체하는 방향이 아니라, `tref`를 main으로 두고 auxiliary tower를 작게, gated/residual하게 붙이는 방향이어야 한다.

## Already Tried N-tower / Fusion Work

### 1. HyFormer WuKong Two Tower

Result:

```text
HyFormer WuKong two tower: 0.800326
HyFormer 10 epoch:         0.813109
HyFormer tref:             0.817137
```

기존 구조:

```text
hy = HyFormerTower(tokens)
wk = WuKongTower(tokens)

fused = [hy ; wk ; hy * wk ; |hy - wk|]
logit = MLP(fused)
```

분석:

- simple concat/product fusion이 main representation을 과하게 흔든 가능성이 크다.
- WuKong tower가 HyFormer와 보완적인 signal을 주기보다 noise 또는 overfit source로 작동했을 수 있다.
- 이 결과만 보면 “큰 auxiliary tower + free MLP fusion”은 위험하다.

### 2. hyf_wk_bifusion

Status:

- `hyf_wk_bifusion` 폴더에 구현 완료.
- 기존 `hyformer_wukong` simple fusion을 FinalMLP-style bilinear fusion으로 교체.
- 실행 시 HyFormer/tref용 extra args가 들어와 argparse 에러가 났고, compatibility-only args를 추가해 무시하도록 수정했다.

현재 fusion:

```text
hy_n = LN(hy)
wk_n = LN(wk)

interaction_h = <P_h(hy_n), P_w(wk_n)>  # low-rank multi-head bilinear

logit = Linear(hy_n)
      + Linear(wk_n)
      + scale * Linear(interaction)
```

의도:

- simple concat MLP보다 tower별 logit과 cross interaction을 분리한다.
- bilinear branch scale을 작게 시작해서 auxiliary interaction이 초반부터 main tower를 망치지 않도록 한다.

주의:

- 이 실험은 `HyFormer tref` 기반이 아니라 기존 compact `hyformer_wukong` 기반이다.
- 따라서 성능이 좋아도 최종 best 후보라기보다는 “bilinear fusion 방식 자체가 simple fusion보다 나은가”를 보는 실험이다.

## Performance-Oriented Priority

성능을 크게 올리려면 직접 ensemble이 불가능한 조건에서, 모델 내부에 ensemble-like diversity를 만드는 N-tower가 필요하다. 다만 지금까지 결과상 auxiliary tower는 약하게 쓰는 것이 맞다.

우선순위는 다음과 같다.

```text
1. tref main + seq-only query auxiliary
2. tref main + DCNv2-lite auxiliary
3. tref main + seq-only query aux + DCNv2-lite aux
4. tref main + bilinear fusion variants
5. WuKong auxiliary 재시도는 bilinear 결과가 좋을 때만
```

## Candidate Experiments

### Experiment A: tref + seq-only query dual tower

가장 먼저 할 실험.

배경:

- Toss ablation에서 `query_seq_only`가 best였다.
- Submission에서는 `tref`가 best였다.
- 둘의 차이는 query generation에 user/item summary를 얼마나 직접 넣느냐다.

구조:

```text
Tower A: tref query
q_i_a = FFN_a([F_u ; F_i ; S_i])

Tower B: seq-only query
q_i_b = FFN_b([S_i])

Option 1: query-level gate
q_i = gate_i * q_i_a + (1 - gate_i) * q_i_b

Option 2: logit-level residual
logit = logit_tref + alpha * gate * logit_seq_only
```

추천 구현:

```text
query-level gate first
```

이유:

- tower를 완전히 두 벌 돌리는 것보다 계산량이 작다.
- 같은 HyFormer block을 공유하면서 query만 다양화할 수 있다.
- `S_i`가 강한 signal이라는 기존 결과와 직접 연결된다.

성공 기준:

```text
submission AUC > 0.817137
```

결과별 다음 단계:

- If `+0.002` 이상: 이 구조를 main으로 두고 seed/regularization 실험.
- If 비슷함: logit-level residual fusion으로 재시도.
- If 하락: seq-only는 auxiliary로도 noise. DCNv2-lite로 이동.

### Experiment B: tref + DCNv2-lite auxiliary

두 번째 우선순위.

배경:

- DCNv2 1 epoch submission AUC가 0.790978로 단순 tabular model 중 강하다.
- HyFormer는 sequence/query 중심이고, DCNv2는 non-seq sparse cross에 강한 inductive bias가 있다.
- WuKong보다 가볍고 안정적일 가능성이 높다.

구조:

```text
h_tref, logit_tref = HyFormerTref(x)
h_dcn, logit_dcn = DCNv2Lite(non-seq tokens)

gate = sigmoid(MLP([h_tref ; h_dcn]))
logit = logit_tref + alpha * gate * logit_dcn
```

초기 설정:

```text
alpha_init = 0.05 or 0.1
DCNv2 depth = 2
cross layers = 2
dcn hidden = 64 or 128
```

중요:

- `logit_tref`를 main으로 유지한다.
- concat 후 큰 MLP로 바로 classification하지 않는다.
- auxiliary contribution은 residual로 제한한다.

성공 기준:

```text
submission AUC > 0.817137
```

결과별 다음 단계:

- If 개선: DCNv2 depth/cross layer/grid를 작게 탐색.
- If train AUC만 오르고 submission 하락: alpha 더 낮추거나 gate dropout 추가.
- If 전반 하락: tabular auxiliary도 현재 HyFormer와 중복. Experiment A로 복귀.

### Experiment C: tref + seq-only aux + DCNv2-lite aux

세 번째 단계. A 또는 B 중 하나가 개선될 때만 진행.

구조:

```text
logit = logit_tref
      + alpha_seq * gate_seq * logit_seq
      + alpha_dcn * gate_dcn * logit_dcn
```

추천 초기화:

```text
alpha_seq = 0.05
alpha_dcn = 0.05
```

주의:

- A/B 둘 다 개선이 없으면 C는 하지 않는다.
- 보조 tower를 두 개 붙이면 train AUC는 오르기 쉽지만 submission 일반화는 나빠질 수 있다.

결과별 다음 단계:

- If 개선: alpha/gate/dropout ablation.
- If A 또는 B보다 낮음: best single auxiliary만 유지.
- If train만 상승: residual scale을 더 낮추고 auxiliary dropout 강화.

### Experiment D: tref + bilinear fusion

`hyf_wk_bifusion` 결과가 simple fusion보다 좋을 때 진행.

목적:

- bilinear fusion이 N-tower fusion 방식으로 유효한지 확인.
- 단순 concat MLP보다 interaction을 구조적으로 제한하는 것이 좋은지 확인.

후보:

```text
1. tref + DCNv2-lite + bilinear fusion
2. tref + seq-only + bilinear fusion
3. tref + tabular MLP + bilinear fusion
```

형태:

```text
logit = Linear(h_main)
      + Linear(h_aux)
      + scale * Bilinear(h_main, h_aux)
```

성공 기준:

- `hyf_wk_bifusion > HyFormer WuKong two tower 0.800326`이면 fusion 방식 자체는 개선.
- `tref + bifusion aux > 0.817137`이면 최종 후보.

결과별 다음 단계:

- If `hyf_wk_bifusion`도 0.80 근처: WuKong tower 자체가 문제. WuKong 계열 중단.
- If simple fusion보다 개선되지만 tref 미만: bilinear fusion은 유효하지만 tower 선택이 문제. DCNv2-lite와 결합.
- If tref 초과: bilinear fusion을 N-tower 기본 fusion으로 채택.

### Experiment E: tref + WuKong auxiliary retry

낮은 우선순위.

진행 조건:

```text
hyf_wk_bifusion이 simple HyFormer WuKong two tower보다 명확히 좋아야 함
```

추천 형태:

```text
logit = logit_tref + alpha * gate * logit_wukong
```

하지 말아야 할 형태:

```text
logit = MLP([h_tref ; h_wukong ; product ; diff])
```

이 형태는 이미 `HyFormer WuKong two tower`에서 좋지 않았다.

## Stop Conditions

다음 조건이면 해당 방향을 중단한다.

```text
1. train AUC만 크게 상승하고 submission AUC 하락
2. HyFormer 10ep 0.813109보다도 낮음
3. runtime이 tref 대비 크게 증가하는데 AUC gain이 +0.001 미만
4. auxiliary tower가 LightGCN/u-init처럼 user/click shortcut에 의존
```

특히 다음 계열은 우선 중단:

```text
LightGCN / clicked graph
user-click init
CNN sequence summary
large free-form concat fusion
```

## Recommended Execution Order

### Step 0: Finish current bifusion sanity check

Run:

```bash
./run.local.sh hyf_wk_bifusion 6 toss --num_epochs 10 --batch_size 256
```

Expected interpretation:

- If `> 0.800326`: bilinear fusion improves over simple WuKong fusion.
- If still around `0.800` or lower: WuKong auxiliary is not worth prioritizing.

### Step 1: Build tref_seqdual

Highest expected value.

```text
base = HyFormer tref
aux = seq-only query generator
fusion = query-level gate
```

Run submission if local does not show obvious failure.

### Step 2: Build tref_dcn_aux

```text
base = HyFormer tref
aux = DCNv2-lite
fusion = residual gated logit
```

Initial alpha should be small.

### Step 3: Combine only winners

Only if Step 1 or Step 2 improves:

```text
tref + seqdual + dcn_aux
```

### Step 4: Apply bilinear fusion to the winning auxiliary

Only if current `hyf_wk_bifusion` indicates bilinear fusion is better than simple fusion.

## Current Best Bet

Most likely to improve submission:

```text
HyFormer tref + seq-only query auxiliary with query-level gate
```

Second best:

```text
HyFormer tref + DCNv2-lite residual logit tower
```

Risky:

```text
HyFormer tref + WuKong full auxiliary
```

Reason:

- The strongest repeated signal is simple `MeanPool(Seq_i)` in query generation.
- `tref` is already best, so main path should stay intact.
- N-tower should add diversity only through a controlled gate/residual, not replace the classifier with a large fusion MLP.
