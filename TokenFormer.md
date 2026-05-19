# TokenFormer Review: HyFormer / UniMixer 비교 관점

## 요약

TokenFormer는 multi-field feature interaction과 sequential recommendation을 하나의 homogeneous Transformer backbone으로 통합하려는 논문이다. 핵심 문제의식은 단순히 non-sequence field token과 sequence token을 같은 Transformer에 넣으면, 정보량이 낮거나 low-rank인 non-sequence feature가 sequence representation까지 붕괴시키는 Sequential Collapse Propagation(SCP)이 발생한다는 점이다.

TokenFormer의 해결책은 두 가지다.

- [ ] Bottom-Full-Top-Sliding(BFTS): shallow layer에서는 full attention으로 feature/sequence/target 간 global interaction을 만들고, deep layer에서는 shrinking sliding-window attention으로 sequence refinement에 집중한다.
- [ ] Non-Linear Interaction Representation(NLIR): attention output에 one-sided multiplicative gate를 적용해 linear attention만으로 생기는 rank collapse를 줄인다.

현재 우리 HyFormer 실험과 연결하면, TokenFormer가 주는 가장 중요한 시사점은 “sequence와 static feature를 무작정 더 많이 섞으면 안 된다”는 것이다. 우리 결과에서도 sequence summary는 중요했지만, tref+NS, CNN, tower fusion처럼 interaction을 복잡하게 만든 실험은 baseline/tref보다 낮았다. 따라서 TokenFormer식 적용은 전체 구조를 갈아엎기보다 HyFormer block 안에서 sequence-token 보호 장치를 넣는 쪽이 현실적이다.

## 논문의 문제의식

추천 모델은 크게 두 흐름으로 발전했다.

- multi-field feature interaction: user/item/context sparse feature 간 상관관계 모델링
- sequential recommendation: user behavior sequence의 시간적 interest evolution 모델링

최근 모델들은 두 흐름을 하나의 backbone에 넣으려 하지만, TokenFormer는 naive unified Transformer가 실패할 수 있다고 본다. non-sequence field는 low-cardinality, sparse, redundant feature가 많아 representation이 낮은 유효 차원에 몰리기 쉽다. 이 token들이 sequence token과 반복적으로 attention하면, sequence representation도 low-rank 방향으로 끌려가고, sequence의 dimensional robustness가 망가진다.

이 현상을 논문은 Sequential Collapse Propagation(SCP)라고 부른다. RankUp의 representation collapse와 비슷하지만, TokenFormer는 특히 “non-sequence feature가 sequence feature를 오염시키는 전파 경로”에 초점을 둔다.

## TokenFormer 구조

### 1. Unified Token Stream

TokenFormer는 non-sequential field feature, behavior sequence, target feature를 하나의 token stream으로 만든다.

```text
X_0 = [F tokens ; sequence tokens ; target tokens ; separators]
```

이 구조의 목표는 다음 interaction을 하나의 backbone에서 모두 처리하는 것이다.

```text
F <-> F
T <-> T
V <-> V
T <-> V
F <-> T
F <-> V
```

여기서 `F`는 static/multi-field feature, `T`는 behavior sequence, `V`는 target item/ad feature다. HyFormer와 비교하면, HyFormer는 sequence domain별 query decoding을 통해 sequence를 분리해서 다루고, TokenFormer는 모든 것을 하나의 token stream으로 넣는다.

### 2. BFTS: Bottom-Full-Top-Sliding Attention

BFTS는 layer 위치에 따라 attention pattern을 다르게 둔다.

```text
bottom layers:
  full causal attention
  -> static / sequence / target 간 global interaction 형성

top layers:
  shrinking sliding-window attention
  -> static token을 계속 재참조하지 않고 sequence refinement 집중
```

논문의 해석은 명확하다. shallow layer에서는 static feature가 sequence와 target에 유용한 prior를 준다. 하지만 deep layer에서 static token을 계속 back-attend하면 sequence representation에 noise가 누적되고 SCP가 발생한다. 따라서 깊은 layer에서는 attention window를 줄이고 sequence-local refinement로 전환한다.

이건 우리 HyFormer에 중요한 힌트다. HyFormer는 sequence domain별 cross-attention 후 decoded query와 NS token을 RankMixer/UniMixer block에서 섞는다. 만약 deep block에서 NS token이 sequence query를 계속 강하게 끌고 가면 TokenFormer가 말한 SCP와 비슷한 문제가 생길 수 있다.

### 3. NLIR: Non-Linear Interaction Representation

NLIR은 attention output을 그대로 residual에 더하지 않고, hidden state에서 만든 gate로 modulation한다.

```text
G_l = X_l W_g
I_tilde_l = sigmoid(G_l) * A_l
```

여기서 `A_l`은 attention output이다. 의도는 attention이 만든 linear mixture를 hidden state 기반의 multiplicative gate로 비선형화해, 특정 low-rank 방향으로 representation이 급격히 몰리는 것을 줄이는 것이다.

우리 관점에서는 `CrossAttention` output이나 HyFormer block의 `decoded_q_i`에 gate를 추가하는 방식으로 작게 실험할 수 있다.

```text
decoded_q = CrossAttention(q, seq)
gate = sigmoid(Linear(q or global))
decoded_q = gate * decoded_q
```

이 정도는 CNN/GCN/tower보다 훨씬 작고, sequence representation 보호라는 목적도 명확하다.

## HyFormer와 비교

| 관점 | HyFormer | TokenFormer |
| --- | --- | --- |
| 기본 철학 | sequence domain별 query decoding | 모든 feature/sequence/target을 unified token stream으로 통합 |
| sequence 처리 | domain별 sequence encoder + cross-attention | decoder-only backbone 내 full/sliding attention |
| non-sequence 처리 | user/item NS token | field token이 sequence token과 같은 stream에 들어감 |
| collapse 문제 | 명시적으로 다루지 않음 | SCP를 핵심 문제로 정의 |
| 해결 방식 | global query, RankMixer/UniMixer mixing | BFTS + NLIR |

HyFormer의 장점은 현재 데이터처럼 4개 domain sequence가 명확히 분리된 경우 구조적 prior가 강하다는 점이다. TokenFormer는 더 통합적인 구조지만, 모든 token을 하나로 넣는 만큼 sequence collapse를 막는 attention schedule이 필요하다.

우리 submission 결과를 보면 HyFormer에서 sequence summary는 매우 중요했다.

```text
HyFormer 10 epoch              0.813109
HyFormer wo meanpool(seq)      0.804653
HyFormer tref                  0.817137
HyFormer tref CNN              0.809216
```

즉 sequence를 쓰는 것은 맞지만, 더 복잡한 sequence encoder가 항상 좋은 것은 아니었다. TokenFormer의 BFTS는 “더 복잡한 sequence encoder”가 아니라 “언제 static feature와 섞고, 언제 sequence만 refine할지 정하는 schedule”이므로 실험 가치가 있다.

## UniMixer / UniMax와 비교

UniMixer 계열은 token mixing 자체를 parameterized/learnable하게 만들어 RankMixer의 fixed mixing을 대체하려는 방향이다. TokenFormer는 mixer를 더 일반화하기보다, attention mask와 interaction nonlinearity를 조절해 unified modeling의 부작용을 줄인다.

| 관점 | UniMixer | TokenFormer |
| --- | --- | --- |
| 핵심 변경 | learnable token mixing | layer별 attention receptive field schedule |
| 문제 정의 | fixed mixer의 표현 한계 | sequence collapse propagation |
| sequence 보호 | 명시적이지 않음 | deep layer에서 static token 재참조 제한 |
| 적용 위치 | NS tokenizer / block mixer | sequence attention / cross-domain interaction |
| 우리 실험 후보 | `hyf_unimixer_*` | `hyf_bfts`, `hyf_nlir` |

현재 우리가 만든 `hyf_unimixer_block`은 decoded query + NS token을 더 유연하게 섞는 실험이다. TokenFormer 관점에서는 이게 오히려 static/non-sequence token의 영향력을 키워 SCP를 만들 수도 있다. 따라서 UniMixer block 실험 결과가 나쁘다면, 단순히 mixer가 약해서가 아니라 “무제한 mixing이 sequence를 오염시켰기 때문”일 수 있다.

## 우리 코드에 적용할 실험 제안

### 1순위: HyFormer CrossAttention NLIR Gate

가장 작고 안전한 실험이다. 각 domain의 cross-attention output에 gate를 곱한다.

```text
decoded_q_i = CrossAttention(q_i, seq_i)
gate_i = sigmoid(Linear([q_i, global_i]))
decoded_q_i = gate_i * decoded_q_i
```

또는 더 단순하게:

```text
gate_i = sigmoid(Linear(q_i))
decoded_q_i = gate_i * decoded_q_i
```

기대 효과:

- sequence attention output이 low-rank 방향으로 과도하게 몰리는 것을 완화
- 추가 파라미터 작음
- `tref` 구조와 충돌 작음

### 2순위: NS-to-Query Mixing 제한

현재 HyFormer block은 다음처럼 모든 decoded query와 NS token을 concat 후 mixer에 넣는다.

```text
combined = [decoded_qs ; ns_tokens]
boosted = RankMixerBlock(combined)
```

TokenFormer식으로 보면 deep block에서 NS token이 query token에 계속 섞이는 것이 위험할 수 있다. 따라서 block별로 mixing mask/schedule을 둔다.

```text
block 1:
  query + NS full mixing

block 2:
  query-query 중심 mixing
  NS 영향은 residual 또는 gate로 제한
```

RankMixerBlock은 fixed reshape라 mask 적용이 어렵다. 따라서 이 실험은 `rank_mixer_mode=unimixer` 또는 별도 masked mixer에서 더 자연스럽다.

### 3순위: BFTS-style Sequence Encoder

현재 sequence encoder가 transformer일 때, domain sequence 내부 attention에 BFTS를 넣는다.

```text
lower block:
  full attention over sequence

upper block:
  sliding window attention
```

다만 우리 HyFormer는 sequence encoder depth가 크지 않고, 10 epoch 기준 학습 시간이 이미 길다. 따라서 전체 TokenFormer식 unified stream을 새로 만들기보다, sequence encoder 내부에 sliding window를 넣는 정도가 현실적이다.

### 4순위: Static Token Drop/Stop in Deep Block

TokenFormer는 full attention phase 이후 non-sequence token을 attention computation에서 제거한다. HyFormer식으로는 deep block에서 NS token update는 유지하되, query update에 들어가는 NS 영향만 줄일 수 있다.

```text
block 1:
  next_q, next_ns = mixer([q, ns])

block 2:
  next_q = q_mixer(q only)
  next_ns = ns_mixer(ns or detached q)
```

이 실험은 `tref`가 이미 좋은 상황에서 over-mixing을 줄이는 regularization 역할을 할 가능성이 있다.

## 실험 우선순위

```text
1. hyf_tref_nlir
   - CrossAttention output에 sigmoid gate 추가
   - 가장 작고 TokenFormer 핵심과 직접 연결

2. hyf_tref_block_schedule
   - 1 block은 q+NS mixing, 2 block은 q 중심 mixing
   - SCP 방지 목적

3. hyf_tref_unimixer_masked
   - UniMixerBlock에 query/NS mask 또는 gate 추가
   - hyf_unimixer_block 결과가 나쁘면 특히 확인

4. hyf_bfts_seq_encoder
   - sequence encoder에 full -> sliding attention schedule
   - 구현 비용은 높지만 TokenFormer 구조와 가장 유사

5. full TokenFormer-style unified stream
   - 장기 과제
   - 현재 데이터/코드에서는 구조 변경이 커서 우선순위 낮음
```

## 결론

TokenFormer의 핵심은 unified modeling 자체보다 “unified modeling을 안전하게 하는 법”이다. sequence와 static feature를 섞는 것은 성능에 필요하지만, deep layer에서 static token을 계속 재참조하면 sequence representation이 무너질 수 있다.

우리 HyFormer 실험에서는 sequence summary가 성능의 핵심이었고, 동시에 복잡한 fusion/GCN/init은 일반화를 해쳤다. 따라서 TokenFormer에서 바로 가져올 아이디어는 full unified Transformer가 아니라, `NLIR gate`, `NS 영향 schedule`, `deep block에서 sequence 보호`다.

가장 먼저 해볼 만한 것은 `hyf_tref_nlir`이다. 구현이 작고, label leakage가 없으며, 현재 best인 `HyFormer tref`의 장점을 유지하면서 sequence-query representation을 보호할 수 있다.

## 참고 자료

- TokenFormer arXiv: https://arxiv.org/abs/2604.13737
- TokenFormer summary: https://chatpaper.com/paper/268754
- Gist Science summary: https://gist.science/paper/2604.13737
