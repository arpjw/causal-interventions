# M1-M3 implementation status

This record distinguishes implemented software from measured research results. Source remains local and uncommitted. The private GitHub remote is connected but has no source pushed to it.

## Completed implementation

| Area | Implemented evidence | Status |
| --- | --- | --- |
| M1: finite SCMs | Positive binary joint tables, all causal-order factorizations, exact truncated-factorization interventions, query probabilities, independent rational-arithmetic tests | Implemented and validated |
| M2: policies | Bayesian posterior, random and greedy policies, exact finite-horizon query-risk policy with STOP, exact scoring, exhaustive-tree checks | Implemented and validated |
| M3: benchmark engine | Deterministic development/test streams, parameter-disjoint episodes, informative and near-tie acceptance gates, rejection accounting, replay-verified manifests, full-horizon opportunity audit | Implemented and validated |
| M3: study freeze | Locked schema, source/prompt/analysis hashes, count rationale, and environment-verification machinery | Implemented; first 360-episode test frozen and completed |

The fixed primary prompt renderer and paired analysis contract are also implemented. Optional local model adapters and a development runner now exist. A paid RunPod A40 pilot executed the pinned Qwen3-32B-AWQ model on 18 development episodes. JSON constrained decoding yielded zero parse failures and a +0.1495-nat paired model-minus-random loss difference on this exploratory sample. The independent locked test result is reported below.

## Validation evidence

- The full local `unittest` suite passes, including exact SCM checks against independent rational arithmetic, finite-horizon policy trees, split isolation, manifest tampering, prompt private-state invariance, and CLI generate/validate/audit/freeze smoke paths.
- The tracked `configs/benchmark-v0.json` is a small budget-one development configuration. `configs/benchmark-factorial-v0.json` exercises two, three, and six candidate worlds; budgets one, two, and three; and informative and near-tie decision regimes. A 10-per-stratum candidate pilot produced 360 episodes across development and test; it is explicitly unfrozen and ignored by Git.
- In that model-free pilot, the mean full-horizon random-to-exact opportunity gap across the nine informative development strata was about 0.00385 nats; the near-tie mean was about 0.00000557 nats. These numbers describe the selected synthetic generator, not model performance.
- A separate realized-outcome check on 90 informative development episodes gave random-minus-exact mean 0.00652 nats and sample standard deviation 0.04264 nats. This noisy pilot is not a power calculation for a language model, but it shows why a very small locked test set would be hard to interpret.

## Frozen evaluation

The [count rationale](test-count-rationale-v1.md) fixed 20 test parameter draws per stratum. The factorial configuration balances candidate count, budget, and decision-value regime, but does not balance weak versus strong causal effects. No candidate test episode was used for prompt or decoder tuning. The freeze command hashed the source, prompt, analysis, and written count rationale.

That sequence was completed under the remote Python 3.12 environment. The 360-episode locked test had zero parse failures and a +0.262067-nat model-minus-random loss difference, with a 95% cluster bootstrap interval of [+0.163531, +0.380312]. The RunPod pod was deleted after artifact collection. See [the locked result](QWEN32_LOCKED_RESULTS.md) for the full evidence and limitations. The local Python 3.14 and remote Python 3.12 interpreters produce slightly different floating-point manifest bytes, so validate this specific lock in Python 3.12 or use the archived source and saved ledger for independent verification.
