# href_init Experiment List

Ordered by expected impact, while keeping dependent checks in the order needed to interpret results:

- [ ] `baseline_href`: current href/tref global construction without user init.
- [ ] `click_user_init_fusion`: train-click user init fused into `F_u`.
- [ ] `click_user_init_fusion + init_dropout`: add dropout/noise to the click init before fusion.
- [ ] `click_user_init_fusion + init_gate`: learn how much to trust click init per row/user.
- [ ] `click_user_init_fusion + init_gate + init_dropout`: strongest expected variant if click init is valid.
- [ ] `seq_user_init_fusion`: sequence-only user init fused into `F_u`.
- [ ] `hybrid_user_init_fusion`: combine click init and sequence init, then fuse into `F_u`.
- [ ] `hybrid_user_init_fusion + init_gate`: gated version of the hybrid init.
- [ ] `click_cluster_user_init_fusion`: SG-URInit-style clicked item aggregate plus cluster aggregate, fused into `F_u`.
- [ ] `seq_cluster_user_init_fusion`: SG-URInit-style sequence aggregate plus cluster aggregate, fused into `F_u`.
- [ ] `click_user_init_global`: train-click user init appended directly to `global_i`.
- [ ] `seq_user_init_global`: sequence-only user init appended directly to `global_i`.
- [ ] `init_stopgrad`: use precomputed user init as a fixed prior without backpropagating into init embeddings.
- [ ] `cold_user_fallback`: explicit fallback for users without train-click history.

Short priority:

```text
Most promising: click_user_init_fusion + init_gate + init_dropout
Safest sanity check: seq_user_init_fusion
Highest leakage risk: click_user_init_global or any graph propagation from clicked labels
```

# Goal

Current href/tref global construction is:

```text
U' = Refiner(U_NS)
I' = Refiner(I_NS)

F_u = MeanPool(U')
F_i = MeanPool(I')
S_i = MeanPool(Seq_i)

global_i = [F_u ; F_i ; S_i]
q_i = FFN_i(global_i)
```

The SG-URInit-inspired direction is to add a user-level semantic prior `G_u`, but avoid the failure mode seen in LightGCN:

```text
train row positive label graph -> train AUC very high
test labels unavailable -> test AUC collapses
```

So `G_u` must be built only from information that is available at inference time:

- Allowed: train-period historical interactions, provided domain sequence features, user/item features.
- Not allowed: validation/test row labels, current row label, any graph edge derived from target labels outside the allowed training cutoff.

# Candidate Integration Points

## 1. Fuse into user summary

Preferred first direction:

```text
G_u = UserInit(user_id or row sequence)
F_u_init = FFN([F_u ; G_u])

global_i = [F_u_init ; F_i ; S_i]
q_i = FFN_i(global_i)
```

This keeps the href global structure mostly unchanged. It treats `G_u` as a user prior, not as an extra sequence-specific signal.

Expected behavior:

- More stable than appending many new tokens or running GNN propagation.
- Less risk of overfitting than a large graph module.
- Consistent with previous results where simple summaries generalized better than CNN/extra NS variants.

## 2. Append to global context

More direct but slightly riskier:

```text
G_u = UserInit(user_id or row sequence)

global_i = [F_u ; F_i ; S_i ; G_u]
q_i = FFN_i(global_i)
```

Expected behavior:

- Lets query generator use user init directly.
- Increases global dimension and may overfit more easily.
- Useful as an ablation to check whether fusion into `F_u` is too restrictive.

# User Init Sources

## A. Sequence-only user init

Build `G_u` from the provided `domain_*_seq_*` features.

```text
G_u_seq = Pool(Embed(domain sequence tokens))
```

Possible pooling:

- Mean over all valid sequence tokens.
- Per-domain mean, then mean over domains.
- Per-domain mean, then small FFN.

Pros:

- No label usage.
- Available for train/valid/test in the same format.
- Lowest leakage risk.

Cons:

- May overlap with `S_i`, because href already uses sequence mean in each query.
- Still useful because `G_u_seq` is user-level global prior, while `S_i` is query-specific local sequence summary.

Recommended first experiment:

```text
seq_user_init_fusion
```

## B. Train-click user init

Build `G_u` from items clicked by the user in the training split only.

```text
G_u_click = Aggregate(ItemEncoder(clicked items of user u in train split))
```

Important leakage rule:

```text
Do not use validation/test labels.
Do not use the current row label.
For offline validation, ideally use only interactions earlier than the validation cutoff.
```

Pros:

- Closest to SG-URInit.
- Uses high-signal positive preference history.

Cons:

- More leakage-prone if split/cutoff is mishandled.
- Cold or sparse users need fallback.
- Can reproduce the LightGCN failure if target labels leak into the init table.

Recommended implementation:

```text
ItemEncoder(item_int_feats, item_id embedding optional) -> e_i
G_u_click = mean or inv-degree weighted mean of e_i over train-clicked history
```

## C. Cluster-enhanced user init

SG-URInit-style denoising:

```text
G_u_raw = Aggregate(item representations)
G_u_cluster = Aggregate(nearest_cluster_center(item representations))
G_u = (1 - alpha) * G_u_raw + alpha * G_u_cluster
```

Suggested `alpha`:

```text
0.01, 0.05, 0.1
```

Pros:

- Denoises noisy item/token representations.
- Keeps user prior compact.

Cons:

- KMeans preprocessing adds complexity.
- If item representations are weak, cluster centers may blur useful preference signals.

Start with:

```text
click_cluster_user_init_fusion
seq_cluster_user_init_fusion
```

## D. Hybrid user init

Combine sequence-based and click-based priors:

```text
G_u = beta * G_u_click + (1 - beta) * G_u_seq
```

or:

```text
G_u = FFN([G_u_click ; G_u_seq])
```

Suggested `beta`:

```text
0.25, 0.5, 0.75
```

Pros:

- Click history captures explicit positive preference.
- Sequence features capture broader behavior and are available per row.

Cons:

- More moving parts.
- Should be tested only after single-source init is understood.

# Regularization Variants

## init_dropout

Apply dropout to `G_u` before fusion:

```text
G_u_reg = Dropout(G_u)
F_u_init = FFN([F_u ; G_u_reg])
```

Suggested dropout:

```text
0.1, 0.2, 0.3
```

Reason:

- Prevents the model from relying too strongly on historical clicked prior.
- Especially important for click-based init.

## init_stopgrad

Use `G_u` as a fixed prior:

```text
G_u = stop_gradient(G_u)
```

Reason:

- SG-URInit is training-free at construction time.
- Fixed init can reduce overfitting to train labels.

In this codebase, if `G_u` is precomputed and loaded as a buffer, this is naturally satisfied.

## init_gate

Learn how much to trust `G_u`:

```text
gate = sigmoid(FFN([F_u ; G_u]))
F_u_init = gate * Project(G_u) + (1 - gate) * F_u
```

Reason:

- Useful for users whose historical init is noisy or missing.
- Safer than always concatenating a strong click prior.

# Cold User and Missing History

For users with no clicked history:

```text
G_u = global mean user init
```

or:

```text
G_u = sequence-only init if available
```

Priority fallback:

```text
train-click init exists -> use click init
else sequence init exists -> use sequence init
else global mean init
```

# Recommended Order

## Stage 1: Establish Baseline

- [ ] `baseline_href`

Decision:

- This is the reference for train AUC, validation AUC, submission AUC, and runtime.

## Stage 2: High-Impact Click Init

- [ ] `click_user_init_fusion`
- [ ] `click_user_init_fusion + init_dropout`
- [ ] `click_user_init_fusion + init_gate`
- [ ] `click_user_init_fusion + init_gate + init_dropout`

Decision:

- If train AUC jumps above `0.90` but validation/submission does not improve, click init is too label-like or leakage-prone.
- If validation/submission improves without train AUC explosion, click init is the strongest direction.
- If gate helps, keep user init as an adaptive prior rather than unconditional concat.
- If dropout helps, the model was over-relying on clicked history.

## Stage 3: Safe Sequence Init

- [ ] `seq_user_init_fusion`

Decision:

- If sequence init improves AUC, user-level prior is useful even without clicked labels.
- If click init failed but sequence init helps, avoid label-derived priors and continue with sequence-only variants.

## Stage 4: Hybrid

- [ ] `hybrid_user_init_fusion`
- [ ] `hybrid_user_init_fusion + init_gate`

Decision:

- Use only if click init and sequence init each show usable signal.
- If gated hybrid wins, keep both sources but let the model downweight noisy histories.

## Stage 5: Cluster Denoising

- [ ] `click_cluster_user_init_fusion`
- [ ] `seq_cluster_user_init_fusion`

Decision:

- Run only after raw click or sequence init has a positive signal.
- If cluster variant improves, item/token representation is noisy and semantic smoothing helps.
- If cluster variant drops, raw feature summary is already better than cluster smoothing.

## Stage 6: Riskier Global Append

- [ ] `click_user_init_global`
- [ ] `seq_user_init_global`

Decision:

- Run only if fusion into `F_u` is positive and we want to test whether direct query access helps.
- If global append is worse than fusion, keep user init inside the user summary path.

## Stage 7: Robustness Utilities

- [ ] `init_stopgrad`
- [ ] `cold_user_fallback`

Decision:

- `init_stopgrad` is useful if train AUC rises too fast or the init table is trainable.
- `cold_user_fallback` is required before serious submission if many users have no train-click history.

# Metrics to Watch

Primary:

```text
submission AUC
validation AUC
```

Overfitting indicators:

```text
train AUC - validation AUC gap
train AUC - submission AUC gap
```

Red flags:

- Train AUC rises above `0.90` while validation/submission does not improve.
- Click-init variant behaves like old LightGCN: very high train AUC and poor test AUC.
- Large gains on validation but no gain on submission, suggesting split leakage or unstable prior.

# Current Best Guess

Most likely useful and safe:

```text
click_user_init_fusion + init_gate + init_dropout
seq_user_init_fusion
hybrid_user_init_fusion + init_gate
```

Most risky:

```text
click_user_init_global
large train-click graph propagation
any method that rebuilds user init using validation/test labels
```

The first implementation should avoid GNN propagation entirely. Treat SG-URInit as a compact precomputed user prior and inject it into href's `F_u` path.
