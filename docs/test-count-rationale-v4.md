# Action-choice study v4: prospective mechanism design

Status: protocol prepared before any v4 GPU development or test response. The frozen test must be run without changing its prompt, conditions, metric, count, or model choices after the first v4 test call.

## Question and score

The v3 unaided 32B, 72B, and 235B conditions each returned `do(Y=0)` for all 450 cases, while exact public-input action values yielded 299/300 optimal informative choices in both assisted 32B and 72B conditions. This study tests whether the fixed-choice failure is sensitive to action IDs and presentation, whether intermediate likelihoods are sufficient to help, and whether a large thinking model can select informative interventions from raw public SCM inputs. The v3 test remains closed to prompt selection.

The action-only score is the **prior-averaged one-step Bayes risk**, in nats, of the selected experiment. Lower is better. Invalid output and `STOP` receive initial no-experiment risk. The uniformly random allowed action is the primary reference; the exact finite policy is a positive control. The model gets one action call per episode. The predeclared confirmatory comparison is the **235B thinking condition minus uniform random risk** on the 300 informative cases. Its 95% cluster-bootstrap interval must lie below zero to establish better-than-random action choice under this protocol. Other condition contrasts are mechanism diagnostics with descriptive paired intervals, not independently confirmatory claims.

## Fresh generator and count

Use `configs/benchmark-action-v4.json`, seed 413927. The 486 episodes comprise 12 development and 150 test cases in each of two informative strata and one near-tie control. Informative strata have two or three observationally equivalent candidate orders; each has an opportunity gap of at least 0.01 nats and an action gap of at least 0.01. Each informative test stratum accepts exactly 50 episodes per optimal semantic action. Test and development joint parameters are disjoint. Near-tie cases have action gaps at most 0.0001 nats and are secondary.

This is a precision choice based on v3's 300 informative cases, not a power guarantee. V3's registered count rationale estimated a roughly 0.0031 to 0.0035-nat 95% half-width for a similar model-minus-random variance. The v4 model-free audit found mean random-to-oracle opportunity gaps 0.028600 and 0.025792 nats in the two informative test strata, with respective minima 0.010012 and 0.010082. The near-tie test mean is 0.00002590. These values use no v4 model response. Analyze 300 informative cases equally weighted by independent joint-parameter draw; report near-tie separately. Use 2,000 paired bootstrap replicates, seed 314159, to estimate model-minus-random intervals. Report selected labels and semantic actions, invalid rates, exact-policy gap closure, and per-stratum metrics.

## Prespecified conditions

Use one pinned `Qwen/Qwen2.5-32B-Instruct-AWQ` checkpoint, revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, for these four conditions on the same GPU and inference stack:

1. `literal_raw`: raw SCM input and fixed literal intervention IDs, preserving the v3 presentation feature on fresh data.
2. `opaque_raw`: same raw SCM input, but A/B/C IDs are deterministically assigned to a shuffled intervention map per episode. The map remains visible in the task. Choice IDs and semantic actions are both logged.
3. `opaque_likelihood`: identical opaque map and public task, plus each candidate world's query probability and full 8-outcome distribution for every allowed experiment. These are computed only from public inputs and do not include action-risk values.
4. `opaque_values`: identical opaque map and public task, plus exact public-input expected-risk values for each allowed action. This is a computational positive control, not unaided reasoning.

Use pinned `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8`, revision `f07f63f2bbd7540917118ebdf3812696ef303b03`, in `opaque_thinking`, with the **same opaque raw public task and action mapping**. This checkpoint differs from Qwen2.5 in architecture and training; a contrast with 32B is not a pure size or reasoning-budget effect. It is a capability test. The 235B thinking model receives at most 2,048 output tokens, with 1,536 reserved for thinking and at least 256 for the constrained final JSON action. The Qwen2.5 direct-JSON conditions receive at most 96 output tokens. For matched Qwen2.5 conditions use temperature 0.7, top-p 0.8, top-k 20, prompt-hash seed, and a strict action enum. The thinking condition uses temperature 0.6, top-p 0.95, top-k 20 and a constrained final action.

The development split may check loading, prompt length, valid action parsing, output budget, throughput, and affordability. Fix adapter bugs **before** the test lock. Development responses must not change the prespecified condition list, fresh test count, score, or decision rule. No test episode limit, exclusions, prompt tuning, or reruns after malformed output. If a condition cannot finish safely within budget, report it incomplete rather than silently substituting a model or dataset.

## Audit, spend, and stop rules

Persist prompt, raw generation including any reasoning text, presented-label mapping, selected semantic action, model config hash, token counts, and per-case risk. A separate verifier must reconstruct every prompt and independently calculate each action's risk with `BeliefModel` and `one_step_risk`. Validate the entire generator rejection stream on the generating Python version. Across Python/libm versions, require record hashes, factorization and scores within 1e-10 and skip only exact rejection-stream replay when numerical threshold drift makes it impossible. Freeze source, prompt, analysis, and rationale hashes before test calls. Archive v3 source bytes separately so its earlier lock remains reconstructable.

The user authorized a new **$75 total RunPod cap** on 2026-10-13. This includes GPU time and attached storage and uses no local GPU. Before each pod, confirm account funds, availability, hourly rate, disk size, transfer path, and a deadline leaving a spend reserve. For a 235B pod, estimate the full 450-case duration from development throughput before starting test. Stop before $75; collect and independently replay local artifacts and delete every temporary pod. RunPod billing may lag, so track elapsed provisioned time times quoted rate plus storage and report final invoice separately when available.
