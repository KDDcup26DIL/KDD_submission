# RankUp Review: HyFormer / UniMixer 비교 관점

## 요약

RankUp은 MetaFormer 계열 industrial ranker에서 단순히 depth, hidden dimension, sequence length를 키우는 것만으로는 representation capacity가 같이 커지지 않는다는 문제를 다룬다. 논문의 핵심 진단은 deep ranker의 token representation이 깊어질수록 low-rank subspace로 수렴하는 representation collapse이며, 이를 Effective Rank로 측정한다.

RankUp의 관점은 HyFormer나 UniMixer처럼 token mixer 자체를 더 강하게 만드는 방향과 다르다. RankUp은 mixer를 바꾸기보다, mixer에 들어가기 전과 layer를 통과하는 동안 token representation의 rank와 다양성이 유지되도록 입력 tokenization, embedding, global token, pretrained embedding interaction, task token을 설계한다.

현재 우리 실험 결과와 연결하면, RankUp에서 가장 참고할 만한 방향은 다음 순서다.

- [ ] Randomized Permutation Splitting: NS token 생성 방식 개선. 비용이 작고 label leakage 위험이 낮다.
- [ ] Global Token 개선: 현재 HyFormer tref의 성공 지점과 직접 연결된다.
- [ ] Cross Pretrained / Dense Embedding Interaction Token: `user_dense_feats_{61,87}` 같은 pretrained user embedding과 item feature의 interaction token 추가.
- [ ] Multi-Embedding: 효과 가능성은 있지만 파라미터/메모리 비용이 커서 후순위.
- [ ] Task-Specific Token: 현재 단일 label이면 우선순위 낮음. multi-task label이 생기면 가치가 커진다.

## 논문의 문제의식

RankUp은 추천 ranker의 scaling law가 실제로 성능을 올리더라도, 그 성능 향상이 representation capacity 증가와 항상 비례하지는 않는다고 본다. RankMixer 분석에서 token representation의 Effective Rank가 layer를 따라 damped oscillation 형태를 보이고, 깊은 layer에서는 오히려 감소하는 현상이 관찰되었다.

Effective Rank는 singular value 분포의 entropy로 정의된다.

```text
erank(H) = exp(- sum_i p_i log p_i)
p_i = sigma_i / sum_j sigma_j
```

값이 낮으면 표현이 소수 방향에 몰려 있고, 높으면 여러 독립 축에 정보가 분산되어 있다는 뜻이다. RankUp의 핵심 주장은 “파라미터 수를 키우는 것”과 “표현 공간을 실제로 넓게 쓰는 것”은 다르다는 것이다.

이 관점은 우리 실험 결과와도 잘 맞는다. LightGCN, user init, heavy fusion처럼 capacity를 무작정 추가한 실험은 오히려 submission에서 크게 무너졌다. 반면 HyFormer tref처럼 global summary를 단순하고 안정적으로 만든 변형은 가장 좋은 성능을 냈다.

## RankUp의 구성 요소

### 1. Randomized Permutation Splitting

기존 방식은 sparse feature를 의미적으로 묶거나, concat된 embedding을 순서대로 나눠 token을 만든다. 이 경우 서로 강하게 correlated된 feature가 같은 token에 몰리면 token 내부 표현이 low-rank가 되기 쉽다.

RankUp은 sparse feature index를 random permutation한 뒤 split한다.

```text
features = [f_1, ..., f_M]
permuted = [f_sigma(1), ..., f_sigma(M)]
tokens = split(permuted features)
```

의도는 correlated feature를 여러 token에 분산시켜 token 간 redundancy를 낮추고, 초기 representation matrix `H_0`의 effective rank를 높이는 것이다.

HyFormer 관점에서는 `RankMixerNSTokenizer` 또는 우리가 만든 `UniMixerNSTokenizer` 바로 앞의 feature ordering을 바꾸는 실험으로 대응된다. 현재 HyFormer는 user feature 전체를 concat한 뒤 `user_ns_tokens=5`, item feature를 `item_ns_tokens=2`로 나누므로, feature 순서가 token 구성에 직접 영향을 준다.

### 2. Multi-Embedding Representation

일반적인 recommender는 각 categorical feature에 대해 하나의 embedding table을 사용한다. RankUp은 feature마다 여러 independent embedding table을 두어 같은 categorical signal을 여러 geometry에서 표현한다.

```text
e_j = {psi_1(f_j), ..., psi_K(f_j)}
```

장점은 초기 표현의 자유도가 커지고 long-tail sparse feature가 하나의 낮은 차원 embedding에 과도하게 압축되는 현상을 줄인다는 점이다.

다만 우리 환경에서는 파라미터와 GPU 메모리가 이미 빡빡하다. LightGCN, u-init, 3-tower 계열에서 메모리와 일반화 문제가 반복됐기 때문에, Multi-Embedding은 바로 크게 넣기보다 high-cardinality 핵심 feature 몇 개에만 제한적으로 적용하는 것이 맞다.

### 3. Global Token Integration

RankUp은 local token만으로는 깊은 layer에서 전체 context를 안정적으로 유지하기 어렵다고 보고, 전체 feature를 pooling/MLP/FM/DCNv2 등으로 요약한 global token을 token sequence 앞에 붙인다.

```text
g = A(f_1, ..., f_M)
H_0 = [g, e_1, ..., e_T]
```

이 부분은 우리 HyFormer tref 결과와 가장 강하게 연결된다.

현재 tref는 다음 구조다.

```text
U' = Refiner(U_NS)
I' = Refiner(I_NS)
F_u = MeanPool(U')
F_i = MeanPool(I')
S_i = MeanPool(Seq_i)
global_i = [F_u ; F_i ; S_i]
q_i = FFN_i(global_i)
```

제출 결과에서 `HyFormer tref`가 0.817137로 가장 좋았고, `tref+NS`, `tref CNN`, `tref-lite`, `seq init`은 모두 낮았다. 즉 우리 데이터에서는 “global token을 복잡하게 만들기”보다 “user/item/sequence의 안정적인 summary를 query 생성에 쓰는 것”이 더 잘 맞았다.

RankUp의 global token은 HyFormer tref를 더 정당화한다. 다만 RankUp은 global token을 backbone token으로 직접 넣고 모든 mixer layer에서 교환하게 만든다. 우리 HyFormer tref는 global summary를 query 생성에만 사용한다. 다음 실험 후보는 이 차이를 줄이는 것이다.

```text
current tref:
global_i -> q_i 생성

RankUp-style:
global token을 q/ns token set에 append
-> HyFormer block mixer에서 함께 update
```

### 4. Cross Integration of Pretrained Embeddings

RankUp은 pretrained user/item embedding을 단순 concat하지 않고, element-wise product로 interaction prior를 만든다.

```text
e_cross = Proj(z_user_pretrained * z_item_pretrained)
```

이 아이디어는 우리 데이터의 `user_dense_feats_{61,87}`와 관련 있다. 이 feature들은 user embedding 성격을 가진 dense feature이고, item 쪽에는 scalar/category feature가 많다. 완전한 two-tower pretrained item embedding이 없다면 다음처럼 약하게 적용할 수 있다.

```text
z_u = Project(user_dense_feats_61, user_dense_feats_87)
z_i = Project(item_int/item_dense summary)
cross_token = Project(z_u * z_i)
```

주의할 점은 이전 `u init` 실험처럼 clicked label 또는 train positive edge를 강하게 주입하면 test mismatch가 생긴다는 것이다. RankUp의 cross embedding은 label graph가 아니라 pretrained retrieval representation 또는 dense feature interaction이므로, valid/test에서 동일하게 계산 가능해야 한다.

### 5. Task-Specific Token Decoupling

RankUp은 32개 CVR sub-task를 joint optimization하는 multi-task 환경에서 task token을 둔다. 각 task token은 shared backbone을 통과하지만 자기 task tower로만 들어간다.

현재 우리 KDD submission은 단일 binary label 중심이므로 바로 적용할 우선순위는 낮다. 단, `label_type`이 실제로 여러 conversion/action subtype을 의미하고 multi-task로 쓸 수 있다면, task token은 다시 검토할 가치가 있다.

## HyFormer와 비교

HyFormer의 강점은 heterogeneous feature와 multi-domain sequence를 분리해 다루는 구조다. 우리 구현 기준으로는 다음 흐름이 핵심이다.

```text
user/item NS token 생성
domain sequence embedding
global summary로 domain query 생성
query가 각 domain sequence에 cross-attention
decoded query + NS token을 RankMixerBlock으로 mixing
```

RankUp과 비교하면:

| 관점 | HyFormer | RankUp |
| --- | --- | --- |
| 주 관심 | sequence-aware query decoding | high-rank representation 유지 |
| token 생성 | RankMixerNSTokenizer: concat/split | randomized split + multi-embedding |
| global 정보 | tref에서 query 생성용 summary | global token이 backbone에 직접 참여 |
| interaction | query-sequence cross-attention + RankMixer | MetaFormer backbone + richer input tokens |
| collapse 대응 | dropout, sparse reinit, summary 안정화 | erank를 직접 겨냥한 구조적 설계 |

HyFormer는 sequence domain이 있는 데이터에서 장점이 크다. 실제로 우리 결과에서도 `MeanPool(seq)` 제거가 성능을 크게 낮췄다. 반대로 RankUp은 sequence modeling 자체보다 sparse/dense feature token representation의 rank 유지에 초점이 있다.

따라서 RankUp을 HyFormer에 적용할 때는 HyFormer의 sequence 구조를 건드리기보다, NS token과 global token 쪽부터 적용하는 것이 맞다.

## UniMixer / UniMax와 비교

여기서는 사용자가 말한 UniMax를 최근 우리가 검토하고 구현한 UniMixer 계열, 즉 learnable token mixing 기반의 UniMixer/UniMixer-Lite 방향으로 해석한다.

UniMixer의 핵심은 RankMixer의 fixed token mixing을 learnable/parameterized mixing으로 바꾸는 것이다. attention, token mixer, FM 계열을 하나의 mixing framework로 보고, local/global mixing을 학습 가능하게 만든다.

RankUp과 UniMixer의 차이는 명확하다.

| 관점 | UniMixer | RankUp |
| --- | --- | --- |
| 핵심 질문 | token을 어떻게 더 잘 섞을 것인가 | token 표현이 왜 low-rank로 무너지는가 |
| 주 변경점 | mixer matrix / local-global mixing | tokenization, embedding, global/cross/task token |
| 표현 다양성 | learnable mixing으로 간접 개선 | erank를 직접 목표로 구조 개선 |
| 비용 | mixer parameter와 normalization 비용 증가 | multi-embedding/cross token 비용 증가 가능 |
| HyFormer 적용 위치 | block mixer, NS tokenizer | NS split, global token, cross token |

우리 repo에 만든 세 변형은 이 구분과 맞는다.

- `hyf_unimixer_token`: NS tokenizer만 UniMixer 방식으로 교체
- `hyf_unimixer_block`: block 내부 mixer만 UniMixer 방식으로 교체
- `hyf_unimixer`: 둘 다 교체

RankUp 관점에서 보면 `hyf_unimixer_block`은 mixer 개선 실험이고, `hyf_unimixer_token`은 tokenization 개선 실험이다. RankUp은 후자가 더 본질적일 가능성을 시사한다. 즉 block mixer를 더 복잡하게 만드는 것보다, 초기 token들이 덜 correlated되고 더 높은 effective rank를 갖게 만드는 것이 더 중요할 수 있다.

## 우리 실험 결과와의 연결

현재 submission 결과에서 가장 중요한 패턴은 다음이다.

| 모델 | AUC | 해석 |
| --- | ---: | --- |
| HyFormer | 0.813109 | 강한 baseline |
| HyFormer tref | 0.817137 | global summary 개선이 유효 |
| HyFormer tref+NS | 0.810007 | 복잡한 NS 추가는 오히려 손해 |
| HyFormer tref CNN | 0.809216 | sequence encoding 복잡화는 손해 |
| HyFormer tref-lite | 0.812782 | refiner 제거는 best 대비 손해 |
| HyFormer LightGCN | 0.627606 | label/graph mismatch 위험 |
| HyFormer u init | 0.619317 | clicked 기반 init은 severe mismatch |
| HyFormer WuKong bilinear | 0.799282 | auxiliary tower/fusion은 현재 불리 |

이 결과를 RankUp 관점에서 보면, 큰 폭의 성능 향상은 “더 큰 외부 모듈 추가”보다 “표현 공간을 덜 붕괴시키는 입력/token 설계”에서 나올 가능성이 높다.

특히 다음 결론이 중요하다.

1. `tref`가 좋은 이유는 RankUp의 global token과 비슷하게, local token/sequence interaction에 안정적인 전역 방향을 제공했기 때문으로 볼 수 있다.
2. GCN/u-init 실패는 RankUp의 cross embedding과 다르다. RankUp은 test 시점에도 계산 가능한 pretrained representation interaction을 쓰지, label-derived clicked graph를 test에 필요한 feature처럼 쓰지 않는다.
3. UniMixer block만 바꾸는 실험은 성능 개선을 보장하지 않는다. RankUp은 mixer 이전의 token rank가 낮으면 mixer를 키워도 deep layer에서 collapse가 생길 수 있다고 본다.

## HyFormer에 적용할 실험 제안

### 1순위: Randomized NS Splitting

현재 `RankMixerNSTokenizer`는 feature group order대로 embedding을 concat하고 split한다. 이 순서를 fixed random permutation으로 바꾼다.

```text
user field order = fixed_random_permutation(user fids)
item field order = fixed_random_permutation(item fids)
concat -> split -> projection
```

권장 실험:

- `hyf_rankup_split`
- seed 1개부터 시작
- user/item 각각 permutation 적용
- sequence domain은 그대로 유지
- `tref` 구조 유지

예상 장점:

- label leakage 없음
- 메모리 증가 거의 없음
- RankUp에서 비용 대비 효과가 좋은 구성

### 2순위: RankUp-style Global Token in Backbone

현재 tref global은 query 생성에만 쓰인다. global token을 NS token set에 추가해 block 내부 mixer에서도 update되게 만든다.

```text
g_i = FFN([F_u ; F_i ; S_i])
ns_tokens = [g_i ; user_ns ; item_ns ; dense_tokens]
```

주의점:

- domain별 `g_i`를 만들면 token 수가 늘어난다.
- `rank_mixer_mode=full`은 `d_model % T == 0` 제약이 있으므로 `T` 재계산 필요.
- full RankMixer 제약이 깨지면 `ffn_only`, `unimixer`, 또는 token 수 조정이 필요하다.

### 3순위: Cross Dense Interaction Token

`user_dense_feats_61/87`을 user pretrained embedding proxy로 보고, item feature summary와 interaction token을 만든다.

```text
z_u = Project(user_dense)
z_i = Project(item_ns_mean or item scalar summary)
e_cross = Project(z_u * z_i)
ns_tokens = [e_cross ; existing_ns_tokens]
```

이 실험은 `u init`과 달리 clicked label graph를 쓰지 않는다. valid/test에서도 동일하게 계산 가능한 feature-only interaction이므로 submission mismatch 위험이 낮다.

### 4순위: Limited Multi-Embedding

전체 sparse feature에 multi-embedding을 적용하면 파라미터가 급증한다. 우선은 user/item의 핵심 low-cardinality 또는 high-impact feature 일부에만 적용한다.

```text
e = concat(Emb_1(fid), Emb_2(fid))
project back to d_model or emb_dim
```

적용 후보:

- item scalar categorical feature
- user_int scalar 중 cardinality가 작고 안정적인 feature
- sequence id feature는 제외 권장

### 5순위: Effective Rank Diagnostic

RankUp을 제대로 검증하려면 AUC만 보지 말고 layer별 erank를 찍어야 한다.

측정 위치:

- NS tokenizer output
- query generator output
- 각 HyFormer block의 mixer 전/후
- final pooled representation

간단한 진단 지표:

```text
erank_mean per batch
erank_user_ns
erank_item_ns
erank_q_tokens
erank_after_block_k
```

이걸 넣으면 `hyf_unimixer_token`, `hyf_unimixer_block`, `hyf_rankup_split` 중 무엇이 실제로 representation collapse를 줄이는지 볼 수 있다.

## 결론

RankUp이 주는 가장 중요한 시사점은 “mixer를 더 세게 만들기 전에, mixer에 들어가는 token들이 충분히 독립적이고 high-rank인지 확인해야 한다”는 것이다.

우리 결과에서는 이미 이 방향의 근거가 있다. `tref`는 global summary를 안정적으로 넣었을 때 가장 좋은 성능을 냈고, 반대로 GCN/u-init/tower fusion처럼 capacity를 더하는 방식은 일반화가 나빴다. 따라서 RankUp을 HyFormer에 적용한다면, 가장 먼저 할 일은 큰 모듈을 붙이는 것이 아니라 `NS token split`, `global token 위치`, `feature-only cross token`을 정리하는 것이다.

실험 우선순위는 다음이 가장 현실적이다.

```text
1. hyf_rankup_split: fixed random permutation NS split
2. hyf_tref_global_token: tref summary를 backbone token으로 추가
3. hyf_cross_dense: user_dense x item_summary interaction token
4. erank diagnostic 추가
5. limited multi-embedding
```

## 참고 자료

- RankUp arXiv: https://arxiv.org/abs/2604.17878
- RankUp paper summary: https://chatpaper.com/paper/270842
- RankUp archivist review: https://www.paper-archivist.com/reading/2026/rankup-towards-high-rank-representations-for-large-scale-adv/
- UniMixer 관련 요약: https://en.shuziqushi.com/new305976.html
