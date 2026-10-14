# Action-choice study v3: prospective locked design

## Question and estimand

The registered primary question is whether a model's one permitted experiment has lower **prior-averaged one-step Bayes risk** than a uniformly random permitted experiment. For each action, enumerate every outcome under every public candidate SCM, update the public prior exactly, and average the remaining query-prediction risk. The model is scored solely on its selected action. A malformed action or `STOP` receives the initial no-experiment risk and remains in the analysis. This measure integrates over outcome noise and does not depend on the generator's hidden-world draw. The exact one-step policy is a positive control; the uniform-action policy is the reference. Lower nats are better.

The matched size contrast is Qwen2.5-72B-Instruct-AWQ minus Qwen2.5-32B-Instruct-AWQ on the same fresh episodes, prompt, direct-JSON decoding, and inference stack. A negative paired contrast favors 72B. The 32B and 72B assisted conditions receive the exact public-input action-value table and test whether selecting from computed values is easier than deriving them. They are diagnostics, not evidence of unaided causal reasoning. Qwen3-235B-A22B-Instruct-2507-FP8 is a larger-model capability condition. It changes architecture and instruction family, so its contrast is not a parameter-only scaling effect.

## Fresh generator and count

Use `configs/benchmark-action-v3.json`, seed 339814, with one action and three allowed interventions. There are two informative strata with two and three observationally equivalent candidate orders, plus a near-tie two-candidate control. An informative case requires at least 0.01 nats expected random-to-oracle opportunity and at least 0.01 nats worst-to-best action gap. The near-tie case has an action gap at most 0.0001. Candidate worlds exactly share their observational joint. Test and development parameters are disjoint.

Generate eight development and 150 test cases in each stratum. In each informative stratum, cap accepted cases at 50 per optimal action, yielding exactly balanced best-action labels on the 150 test cases. This prevents a fixed-action shortcut. The 300 informative test cases form the primary analysis, and 150 near-tie cases are secondary. Each independent joint-parameter draw has equal weight. No model response on this fresh test may be used to change the prompt, metric, sample count, or conditions.

The count is a precision choice, not a power guarantee. On the *previous* v2 test, the unregistered expected-risk diagnostic had per-episode standard deviations of 0.0272 nats for 32B and 0.0310 for 72B across 100 informative cases each. If v3 variance is similar, 300 informative cases give an approximate normal 95% half-width of 0.0031 to 0.0035 nats for a model-minus-random mean. This planning calculation does not use any v3 model outputs. The fresh model-free v3 audit found mean random-to-exact gaps of 0.02779 and 0.02518 nats in the two informative test strata, with every case above 0.01; the near-tie mean is 0.0000290. Optimal labels are exactly 50/50/50 in each informative test stratum.

Use 2,000 paired bootstrap replicates over independent joint-parameter draws, fixed seed 314159. Report all invalid actions, chosen-action counts, mean model/random/exact risk, paired model-minus-random difference, 95% interval, and gap closure. Evidence of better-than-random unaided action choice requires the model-minus-random interval to be wholly below zero. Evidence of a matched size benefit requires the paired 72B-minus-32B interval to be wholly below zero; no resolved contrast means no established size effect. Report both assisted conditions and the 235B condition as mechanism/capability diagnostics, with their intervals and architecture caveats. No unplanned exclusions or reruns after malformed responses.

## Model and compute controls

Prespecified checkpoints and immutable Hugging Face revisions:

- `Qwen/Qwen2.5-32B-Instruct-AWQ` at `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`.
- `Qwen/Qwen2.5-72B-Instruct-AWQ` at `698703eae6604af048a3d2f509995dc302088217`.
- `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` at `e156cb4efae43fbee1a1ab073f946a1377e6b969`.

Use `prompts/action-v3.txt`, a constrained JSON schema whose `action_id` enum contains only the three allowed actions and `STOP`, temperature 0.7, top-p 0.8, top-k 20, seed from the prompt hash, at most 96 output tokens, and one model call per episode. The two Qwen2.5 conditions share the same software and GPU configuration. The assisted condition changes only the public task's computed action-value field. The larger FP8 condition may require tensor parallelism and a different GPU count; record this separately. Development calls may establish that the adapter loads, returns valid JSON, and fits the declared time/cost envelope. They may not motivate edits to the frozen test prompt or score. Persist exact prompts, raw responses, model configuration hash, action selections, token counts, parse failures, and per-case scores. Verify each completed ledger with a separate BeliefModel action-value replay.

The new RunPod cycle has a **$75 total hard cap**, explicitly chosen by the user on 2026-10-13. Include GPU and attached storage, use no local GPU, and remove temporary pods after collecting and validating artifacts. Do not provision until the account has sufficient funds and a feasible transfer path. Stop before the cap rather than running a partial condition past it. The prior v2 cap and spend belong to a completed cycle.

The generator records its exact Python version. A verifier on that version replays the complete rejection stream; a different Python version checks the serialized hashes, all record identities, factorizations and diagnostics within 1e-12, split disjointness, and count accounting. Tiny libm differences can move a candidate across a strict rejection threshold and make cross-version stream replay impossible. The local generating runtime must pass full replay before any GPU call.
