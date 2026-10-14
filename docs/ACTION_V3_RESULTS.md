# Action-choice benchmark v3: matched scaling and 235B GPU study

Status: **complete**. Five prespecified GPU conditions finished all 450 locked test cases each. Every ledger passed a separate finite-Bayes-risk replay. Both temporary RunPod pods were deleted after artifact collection.

## Question and locked design

Does a model choose an intervention with lower expected remaining uncertainty about an untried intervention than a uniformly random allowed action? The primary score is the **prior-averaged one-step Bayes risk** of the selected action, in nats. For each action, the evaluator enumerates candidate SCMs and possible experimental outcomes, updates the prior, and scores the exact posterior query probability. This removes the realized-outcome noise that limited v2. The model makes one action call and does not submit a forecast in v3. An invalid action or `STOP` receives the no-experiment risk.

The frozen test has 300 informative cases, 150 with two candidate worlds and 150 with three, plus 150 near-tie controls excluded from the primary comparison. In each informative stratum, each of the three allowed actions is optimal in exactly 50 cases. The model-free random-to-exact opportunity is 0.026489 nats on the pooled informative set. The 95% intervals below use 2,000 paired bootstrap replicates over independent joint-parameter draws. A better-than-random unaided action choice required the model-minus-random interval wholly below zero.

The final lock is `37d3fbd0dd84be40878c244b87f6258dda3358f3af242c18e32702b88909c99d` at ignored local `data/action-v3-final-locked.json`. Its [prospective rationale](test-count-rationale-v3.md) records the counts, gates, conditions, and decision rule. Full generator replay passed on the creating Python 3.14.7 runtime. The GPU Python 3.12.3 runtime verified the lock and every record with a 1e-12 numerical tolerance; slight `libm` differences preclude an identical cross-version rejection stream.

## Registered and diagnostic results

| Condition | Mean model risk | Model minus random | 95% interval | Exact best action, informative | Invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 32B AWQ, unaided | 0.136825 | +0.000172 | [-0.002799, +0.003092] | 100/300 | 0/450 |
| Qwen2.5 72B AWQ, unaided | 0.136825 | +0.000172 | [-0.002799, +0.003092] | 100/300 | 0/450 |
| Qwen2.5 32B AWQ, action values supplied | 0.110167 | -0.026486 | [-0.029611, -0.023453] | 299/300 | 0/450 |
| Qwen2.5 72B AWQ, action values supplied | 0.110167 | -0.026486 | [-0.029611, -0.023453] | 299/300 | 0/450 |
| Qwen3 235B-A22B FP8, unaided | 0.136825 | +0.000172 | [-0.002799, +0.003092] | 100/300 | 0/450 |

The mean uniform-random risk was **0.136653** and the exact one-step risk was **0.110164** nats. Positive model-minus-random differences favor random. Each unaided model chose `do(Y=0)` on **all 450 cases**. Because the informative optimal-action labels were balanced, that fixed choice was optimal in exactly one third of primary cases. The matched 72B-minus-32B risk contrast was exactly **0.000000 nats** with interval [0, 0], since the action choices were identical. The 235B model also made those choices, so it did not improve on 72B under this protocol. Neither unaided model met the registered better-than-random criterion.

With the exact action-value table calculated from the same public SCMs, both Qwen2.5 models selected the best action in 299 of 300 informative cases. The sole miss cost 0.001054 nats in that case. Their mean risk was 0.0000035 nats above the exact policy, closing **99.987%** of the random-to-exact gap. The paired assisted-minus-unaided difference was -0.026658 nats, with a descriptive 95% interval [-0.031484, -0.022148]. This is a diagnostic that the models can use supplied values; the table effectively supplies the computational answer and does not establish unaided causal calculation.

In the near-tie control, the mean model-minus-random risk was about +0.0000092 nats for each unaided condition and -0.0000290 nats for each assisted condition. These effects are tiny by design and are not part of the primary claim.

## Model, inference, and audit receipts

The matched conditions used the same action prompt, constrained JSON action schema, temperature 0.7, top-p 0.8, top-k 20, seed derived from the prompt hash, and 96-token output cap on one Secure Cloud H100 SXM. The assisted conditions changed only the computed action-value field in the public task. The 235B condition used the same prompt and decoding with four H200 SXM GPUs and tensor parallel size four. It is a Qwen3 mixture-of-experts model with **235B total parameters and 22B activated**, so its comparison is not an isolated parameter-size effect. The checkpoints were pinned to the revisions in the prospective rationale. The pod image was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with vLLM 0.10.2, Transformers 4.57.3, Hugging Face Hub 0.36.0, and PyTorch 2.8.0+cu128.

| Condition ledger under ignored `results/` | SHA256 of `runs.jsonl` | Model run wall time |
| --- | --- | ---: |
| `v3-test32-base` | `bbfc90871ae714f0ce17f09047a4a3807f7dcdbfa400a56e95b6a68f874a8a7f` | 145.5 s |
| `v3-test72-base` | `ad60833492e6cdddef2fdf8b756fe7f227999984025d6be662785ebd15a57531` | 208.4 s |
| `v3-test32-assisted` | `c93103c4cf3899ba28b46e807bcdb1324bb197a8b57682a21941e3a0c881c2f6` | 152.5 s |
| `v3-test72-assisted` | `c620d6ff2d092222f59a08be94d001172c5af551dd2984282089f84ace4c7474` | 224.7 s |
| `v3-test235-base` | `b90a5feb04bd90ed8b86d16a6bd59f374808eca5bad33098816100fe4cc0539e` | 493.4 s |

Each completed remote ledger passed `scripts/verify_action_v3.py`, which separately recomputes values with `BeliefModel` and `one_step_risk`, checks every response and prompt, and checks the saved summary. Local replay passed for all five ledgers. The post-lock `scripts/verify_action_v3_portable.py` compares embedded action-value floats within 1e-10 across the macOS/Linux runtimes while still requiring nonnumeric prompt fields and risk scores to agree. It did not select conditions or alter the frozen test. All 2,250 model-case records are present. The repository's 58 unit tests pass.

Two pods were created for this cycle: H100 SXM `ht1bvtfomsutxf` at $3.99/hour and four H200 SXM `6ncy09824rft1e` at $21.16/hour, plus storage. Both were deleted and subsequently returned 404; the account's pod list is empty. The $75 hard cap was respected. From the quoted rates and observed creation/deletion window, GPU compute was at most approximately **$8.5**, with only a few cents of [published volume and container storage rates](https://www.runpod.io/pricing) expected on these short-lived pods. RunPod had not posted this cycle's final billing records at report time, so this is an estimate rather than an invoice.

## Interpretation and limits

The fixed `do(Y=0)` choice shows that these unaided direct-JSON conditions did not adapt their experiment to the candidate causal worlds, despite clear model-free opportunities. The action-value intervention nearly eliminates the deficit, pointing to calculation or use of the full SCM description as the bottleneck in this task. This is an inference from the contrast, not a proof of a single internal failure mechanism.

This benchmark has three binary variables, one held-out query form, one action, a fixed action presentation order, known candidate worlds and priors, and synthetic exact SCMs. The strict schema and 96-token direct-answer protocol may favor a simple default action; there was no reasoning trace or external calculator in the unaided arm. The results do not establish a general inability of larger models to reason causally. A follow-up should randomize action labels and order, supply an intermediate public distribution table without action values, and compare longer reasoning or calculator access on a **new locked test**. The existing v3 test should remain closed to prompt tuning.

The private GitHub repository remains `arpjw/causal-interventions`. These local results and code are uncommitted and unpushed under the repository working agreement.
