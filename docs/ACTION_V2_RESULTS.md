# Action-choice benchmark v2: locked GPU evaluation

Status: both prespecified model conditions completed and their ledgers passed independent replay. The temporary RunPod pod was deleted after collection.

## Question and design

The primary question is whether the model chooses experiments that improve its predictions of an untried intervention. For each model's recorded action and observation, the evaluator computes the exact Bayesian posterior forecast from the public candidate SCMs. It scores that forecast against the hidden world's exact query probability. This measures the experiment choice separately from the model's ability to write a calibrated numerical forecast.

The fresh v2 benchmark has 100 informative test episodes, 50 each with two and three observationally equivalent candidate worlds, and 50 near-tie controls. Every episode permits one action from `do(X=0)`, `do(Y=0)`, and `do(Y=1)`. The held-out query is `P(Z=1|do(X=1))`. Random and exact finite-horizon policies use the same hidden world and seeded potential outcomes. The frozen 95% uncertainty interval resamples independent joint-parameter draws with 2,000 bootstrap replicates. A model demonstrates useful action selection only if its paired model-minus-random interval is wholly below zero. The 50 near-tie episodes and submitted model forecasts are secondary.

The lock hash is `b7af1706ddc5d57f62c9a9eb33bf7e735d97518d4d95e1603ebc88a3d0deba75`. The frozen rationale, prompt, source hashes, generator rules, and baseline audit are embedded in the ignored local manifest at `data/action-v2-locked.json`. The 32B and 72B conditions are distinct model families, so differences cannot be attributed to parameter count alone. The informative generator's model-free mean expected random-to-oracle gap was 0.030636 nats. The realized paired random-to-oracle gap was 0.012660 nats for both model runs.

## Registered result

| Condition | Model action + exact Bayes loss | Random loss | Exact policy loss | Model minus random | 95% cluster interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B-AWQ, budgeted thinking | 0.170792 | 0.151655 | 0.138995 | +0.019137 | [-0.001678, +0.041147] |
| Qwen2.5-72B-Instruct-AWQ, direct JSON | 0.161654 | 0.151655 | 0.138995 | +0.009999 | [-0.009741, +0.030169] |

Positive differences favor random. Both intervals cross zero, so neither condition demonstrates useful experiment selection under the registered decision rule. The 72B point estimate is lower than the 32B estimate, but this contrast does not isolate model size and is not a registered model-to-model test.

## 72B direct JSON condition

Qwen2.5-72B-Instruct-AWQ, revision `698703eae6604af048a3d2f509995dc302088217`, completed all 150 locked episodes on a RunPod H100 PCIe with vLLM 0.10.2. No parse failures occurred.

The paired model-minus-random difference is +0.009999 nats with a 95% cluster bootstrap interval of [-0.009741, +0.030169]. Positive values favor random; the interval crosses zero. The realized random-to-exact gap is 0.012660 nats, smaller than the roughly 0.03-nat model-free expected gap. These data do not establish that 72B action selection is better or worse than random.

The submitted forecast metric is worse than random by +0.154529 nats (95% interval [+0.107622, +0.203301]); 103 of 150 submitted forecasts are exactly 0.5. This is a secondary diagnosis of numerical forecasting, not the primary action-choice claim. The near-tie action-conditioned difference is +0.000186 nats with an interval crossing zero.

The complete 72B ledger is in ignored local `results/runpod-action-v2-locked72/`. Its 150 episodes took 290 seconds of measured model inference after load, using 275,198 input and 6,429 output tokens. An independent finite-table replay checked all 150 action-conditioned posterior forecasts, the paired primary score, manifest hash, and source hashes.

## 32B budgeted-thinking condition

Qwen3-32B-AWQ, revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, completed all 150 locked episodes on the same H100. Each logical response allowed up to 384 thinking tokens, followed by a JSON-constrained final continuation if needed. The response-wide cap was 1,536 generated tokens. This budgeted-thinking protocol was selected using development episodes before the v2 lock. It is not an unrestricted thinking-mode result.

The paired action-conditioned model-minus-random difference is +0.019137 nats with a 95% cluster bootstrap interval of [-0.001678, +0.041147]. Its interval crosses zero. One action response chose `do(X=1)`, the forbidden held-out query intervention. The runner recorded a parse failure, consumed that action slot, supplied no observation, and retained the episode in the primary analysis. No forecast response failed to parse.

The submitted forecast metric is worse than random by +0.604249 nats (95% interval [+0.364798, +0.892538]). Of 150 submitted forecasts, 60 are exactly 0.5 and 11 lie at or beyond 0.001/0.999. This large forecast error illustrates why submitted probabilities are secondary in v2. The near-tie action-conditioned difference is -0.000276 nats with an interval crossing zero.

The complete 32B ledger is in ignored local `results/runpod-action-v2-locked32/`. The full condition took 3,608 seconds of wall time, of which 3,546 seconds were measured model inference, and used 642,058 input and 111,453 output tokens across the logical action and forecast requests. The independent finite-table replay checked all 150 action-conditioned forecasts, the paired primary score, manifest hash, source hashes, and the retained invalid action.

## Exploratory action-value diagnostic

After the locked results, we also computed each chosen action's prior-averaged one-step Bayes risk from the generator's exact action table. This removes realized outcome noise but was not registered as the primary metric. On the 100 informative episodes, the mean chosen-action risk minus uniform-random expected risk is +0.006050 nats for 32B and +0.004801 nats for 72B; positive is worse. These descriptive values support caution about action selection, but they do not replace the registered realized-path comparison or its interval.

## Compute and verification

One Secure Cloud H100 PCIe pod (`u858behsmrb0mf`) was created on 2026-10-13 at 22:09:26 UTC. Its catalog GPU rate was $2.89/hour, plus storage, under the approved $25 cap. Both pinned AWQ snapshots ran offline with vLLM 0.10.2, Transformers 4.57.3, Hugging Face Hub 0.36.0, and PyTorch 2.8.0+cu128. The pod was deleted after collecting both ledgers; `get-pod` returned 404 afterward. RunPod billing had posted $1.751 for the first time bucket at deletion, while later usage was still pending. Based on the quoted GPU rate and roughly 98 minutes elapsed, compute was approximately $4.7 before storage, comfortably below the cap; this is an estimate, not a final invoice.

Local validation included 54 unit tests, remote manifest generation and model-free baseline audit, locked source-hash verification before each model run, and independent local replay of all 300 model-episode records using finite-table factorization and Bernoulli KL scoring. The independent check matched each action-conditioned forecast and both primary paired scores to numerical tolerance.

The independent verifier was written after the lock and is outside the frozen inference code. With the ignored local ledgers present, rerun it using:

```sh
python3 scripts/verify_action_v2.py data/action-v2-locked.json results/runpod-action-v2-locked32
python3 scripts/verify_action_v2.py data/action-v2-locked.json results/runpod-action-v2-locked72
```

## Limits

This is a synthetic, fully specified finite-SCM task. It tests behavior under supplied candidate worlds and exact tables; it does not show that a model learns causal structure from real data or that a larger parameter count caused any observed difference. The realized random baseline is noisy even after pairing, so the prespecified interval and observed opportunity gap matter more than a raw rank ordering. The first Qwen3-32B-AWQ locked study used a different generator and an end-to-end forecast primary score; its result is not pooled with v2.
