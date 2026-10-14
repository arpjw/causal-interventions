# Engineering plan: causal experiment design by AI agents

Status: M1-M3 software and freeze machinery implemented. The first 360-episode locked Qwen3-32B-AWQ evaluation is complete and worse than random on the submitted-forecast primary metric. A fresh v2 action-choice benchmark and two H100 model conditions also completed; neither showed a resolved action-choice improvement over random. See [the first locked result](QWEN32_LOCKED_RESULTS.md) and [the v2 locked result](ACTION_V2_RESULTS.md) for measurements and limitations. The existing two-variable simulator is a smoke test.

## 1. Research target

Test whether an AI agent chooses interventions that improve its predictions about **untried** interventions when observational evidence cannot distinguish competing causal worlds.

The unit of evaluation is an episode: a hidden causal world, a known candidate set, a limited intervention budget, and a set of held-out interventional queries. The agent selects experiments, sees their outcomes, then reports probabilities for the queries. The primary claim is about experimentally useful action choice, not observational prediction or verbal explanations.

This is a controlled behavioral study. Success does not establish that a model has an internal causal representation; failure does not establish that it cannot reason causally outside this task family.

## 2. Scope

### Primary study

- Binary structural causal models (SCMs) with three variables and exact probability tables.
- Two to six candidate SCMs per episode, all with exactly the same observational joint distribution.
- One, two, or three single-variable interventions per episode.
- A fixed set of held-out query interventions. The agent cannot run those exact interventions during exploration.
- Full intervention outcomes are observed for variables other than the intervened one.
- At least one open-weight language model, evaluated without training in the first study.
- Random, greedy information-gain, greedy query-risk, and exact finite-horizon Bayesian policies.

### Deferred extensions

- Four-variable worlds, unknown parameters, partial observations, noisy measurements, unequal intervention costs, natural-language stories, and model fine-tuning.
- Real-world causal claims. The synthetic study is a necessary diagnostic, not a substitute for validation on actual interventions.

Do not add these extensions until the primary benchmark and baselines pass their validation gates. A negative primary result is a valid endpoint.

## 3. Mathematical environment

### 3.1 Construct observationally equivalent worlds

Generate a strictly positive joint distribution `J(x, y, z)` over eight binary states. Bound cell probabilities away from zero for numerical stability. For each selected variable ordering, factorize the **same** `J` into conditional probability tables. Example:

```
X -> Y -> Z, with X -> Z:
P(X) P(Y | X) P(Z | X, Y)

Y -> X -> Z, with Y -> Z:
P(Y) P(X | Y) P(Z | X, Y)
```

Both factorizations reproduce `J` exactly, but the truncated factorization after `do(X=x)` or `do(Y=y)` generally differs. An ordering alone is not evidence of a causal direction; it defines a candidate SCM for this experiment. Select candidate sets with nontrivial interventional disagreement and record the exact parameters.

For a world `w`, action `a = do(V=v)`, and observed non-target variables `o`, compute `P_w(o | do(a))` by replacing the target conditional with a point mass and enumerating all eight assignments. Compute every query probability with the same exact enumeration. No Monte Carlo estimate enters the oracle.

The existing `BinaryWorld` pair is retained as an exact unit test. It is not the main benchmark because it has only one useful intervention target for predicting `Y`.

### 3.2 Belief and actions

The hidden world `W` is drawn from a published candidate prior `b_0`, uniform in the primary study. Every policy receives the same candidate descriptions, observational table, allowed actions, budget, and query definitions. This removes unequal knowledge of model parameters as a confound. The true world index is never shown.

After action `a` and outcome `o`, update:

```
b_next(w) = b(w) P_w(o | do(a)) / sum_u b(u) P_u(o | do(a))
```

If the denominator is zero, reject the episode as a generator or implementation failure; do not silently smooth an impossible observation.

Each intervention costs one unit. A `STOP` action ends exploration without spending the remaining budget. Action sets exclude all exact query interventions and exclude interventions on a query outcome variable when that would reveal it trivially. The generator must include episodes with at least two actions with meaningfully different expected value, plus near-tie cases as a separate difficulty stratum.

### 3.3 Prediction and loss

For query `q`, world `w` has exact Bernoulli probability `p_wq`. A Bayesian forecast under belief `b` is `p_bq = sum_w b(w) p_wq`.

Primary loss is mean excess log loss across held-out queries:

```
L(w, p_hat) = mean_q KL(Bernoulli(p_wq) || Bernoulli(p_hat_q))
```

Using the known interventional probability rather than one sampled query outcome removes avoidable evaluation noise. Clip submitted probabilities to `[1e-6, 1 - 1e-6]` for finite reporting, and report the clipping count. Lower is better. Secondary metrics: action value, world-posterior log loss, parse-failure rate, and inference cost. Do not use world-identification accuracy as the sole outcome: some worlds can remain structurally ambiguous while agreeing on the queries.

### 3.4 Exact benchmark policies

Implement four policies behind one interface:

1. **Random:** uniform over allowed actions, with recorded RNG seed.
2. **Greedy world information gain:** maximize `H(W | history) - E_o H(W | history, a, o)`.
3. **Greedy query-risk reduction:** choose the action with largest one-step reduction in expected query loss.
4. **Bayes-optimal finite-horizon policy:** minimize expected terminal query loss by dynamic programming over the remaining budget.

For the exact policy, define `R(b) = sum_w b(w) mean_q KL(Bern(p_wq) || Bern(p_bq))`. Then `V_0(b) = R(b)` and `V_k(b) = min(R(b), min_a sum_o P(o | b, a) V_(k-1)(b_after(a,o)))`, where the first term represents `STOP`. Cache equivalent belief states and use deterministic tie-breaking. Verify the budget-one result by brute-force enumeration and verify that `V_(k+1) <= V_k`. World information gain is intentionally separate from query value: discovering a structural detail that does not change the query is not useful here.

## 4. Episode generation and split

1. Draw a positive joint table from a predeclared parameterized family. Start with bounded log-linear coefficients; normalize exactly with high-precision or ordinary double arithmetic plus strict tolerance checks.
2. Factor it into the chosen candidate SCMs. Enumerate all observational and interventional probabilities.
3. Reject degenerate draws using **predeclared, model-blind** criteria: minimum cell probability, candidate disagreement on at least one query, a minimum gap between the best and a distractor experiment for the informative stratum, and a maximum oracle difficulty. Record rejection counts and reasons.
4. Stratify accepted episodes by effect strength, candidate count, action-value gap, and budget. Draw balanced numbers per stratum.
5. Separate development and locked test sets by **joint-distribution parameter draw**, not just by outcome RNG seed. Freeze test files and hashes before prompt or policy tuning. Use fresh world and outcome seeds within each split.
6. Pair policy evaluations on the same hidden worlds and pre-generated potential outcomes for each `(episode, action, step)` where possible. This lowers variance without changing the target distribution.

The first benchmark should have a small development set for implementation and prompt debugging, then a locked test set large enough for uncertainty estimates across independently generated worlds. Choose exact counts after a local variance pilot with non-model baselines; record them before any test model call. Do not choose counts by looking at the model's test performance.

## 5. Agent interface

Use a versioned, structured request and response. The first model condition gets a compact numeric table, candidate-world labels, allowed actions, remaining budget, and query definitions. No causal graph names that imply an answer beyond the supplied candidate descriptions.

Expected action response:

```json
{"action_id": "do(X=0)"}
```

Expected final response:

```json
{"forecasts": {"P(Z=1|do(X=1))": 0.63}}
```

`{"action_id": "STOP"}` is also valid and proceeds directly to final forecasts. Validate JSON and numeric ranges. In the primary evaluation, do not retry a malformed response: an invalid action consumes one budget unit and reveals no outcome; an invalid final forecast receives `0.5` for each missing or invalid query. Log these failures separately. This preserves equal call budgets and prevents selective exclusion. The parser must not infer the intended action from prose.

The adapter records model identifier and revision, prompt template hash, decoding settings, input/output token counts, latency, raw response path, parsed response, and total cost. Raw traces belong under ignored `artifacts/`; publish only reviewed, non-sensitive summaries. No external service credentials belong in Git.

## 6. Software layout

Keep the exact core dependency-light. Add modules only when their interface is needed:

```
causal_interventions/
  worlds.py          # binary smoke test, then generic finite SCMs
  factorize.py       # joint table -> conditional tables
  interventions.py  # exact truncated-factorization enumeration
  episodes.py        # generation, strata, split manifests
  beliefs.py         # likelihoods and posterior updates
  policies.py        # random, greedy, finite-horizon oracle
  scoring.py         # query loss and paired summaries
  agent.py           # typed request/response contract
  prompts.py         # versioned structured prompt rendering
  runner.py          # deterministic episode state machine
  reports.py         # summaries and figures from frozen records
tests/
docs/
  protocol.md
  ENGINEERING_PLAN.md
configs/              # tracked, non-secret study specifications
data/                 # ignored generated episodes
results/              # ignored run outputs
artifacts/            # ignored raw model traces
```

Use Python 3.10+ and the current `unittest` convention for the exact core. Introduce a package manager and pin dependencies only when model inference or plotting requires them. Expose commands for `generate`, `validate`, `baseline`, `evaluate`, and `report`, each accepting a config path and writing a machine-readable manifest. Never overwrite a completed run directory; create a run ID from config hash, code revision or source hash, model revision, and timestamp.

Every episode record should include schema version, candidate SCM identifiers and parameters, prior, hidden world, observational table, action set, held-out queries, split, generator seed, and validation hashes. Every step record should include belief or agent input, action, sampled outcome, remaining budget, and parse status. Reports must be reproducible from these records without rerunning a model.

## 7. Verification gates

### Gate A: simulator correctness

- Every candidate factorization reproduces its source observational joint within `1e-10`.
- Intervention probabilities normalize to one and match hand-calculated two-variable and selected three-variable cases.
- Observational data have zero likelihood ratio between candidate worlds by construction.
- Interventions that should distinguish candidates do so; negative controls remain uninformative.
- Sampling frequencies approach exact probabilities in a seeded simulation check. Exact enumeration, not sampling, remains the source of truth.

### Gate B: decision core

- Posterior updates preserve normalization and match hand-worked examples.
- Greedy scores match brute-force enumeration for tiny cases.
- Dynamic-programming actions match exhaustive policy-tree search for budgets one and two.
- All policies see identical public information, budget, and outcome draws.

### Gate C: benchmark freeze

- Generator rejection rules, strata, split IDs, seeds, prompts, baseline code, primary metric, and analysis script are versioned and hashed.
- Development and test parameter draws do not overlap.
- At least two useful action choices exist in the informative stratum.
- Random and exact policies show a measurable opportunity gap. If the gap is too small, redesign the generator **before** model evaluation and version the change.

### Gate D: model run

- Run parser and cost-meter smoke tests on development episodes only.
- Verify the exact model revision and maximum calls or tokens. Stop at the approved hard budget cap.
- Run the locked evaluation once per preregistered model and prompt condition; preserve failures.
- Independently regenerate a sample of episode probabilities and final scores from the raw manifest.

### Gate E: report

- Publish world-level paired estimates with uncertainty intervals, stratum plots, action confusion tables, invalid-output rates, and cost.
- Show representative failures selected by predeclared rules, not only hand-picked examples.
- State which gates passed, which failed, and whether any tests were exploratory.

## 8. Analysis and decision rules

Primary comparison: model excess log loss versus random and Bayes-optimal policies on the locked test episodes, paired by world. Report `gap closure = (L_random - L_model) / (L_random - L_oracle)` only when the denominator exceeds the predeclared minimum opportunity gap; report raw losses everywhere. Bootstrap over independently generated joint tables or world pairs, not over repeated prompts from one table.

Evidence of useful experiment selection requires a confidence interval for the paired model-minus-random loss difference entirely below zero and a positive gap closure. A practically strong result would close at least half the random-to-oracle gap; treat that as a descriptive threshold, not a license to change the benchmark. Compare the model with greedy query-risk as well; outperforming random alone is a modest result.

Secondary tests: budget scaling, weak versus strong causal effects, prompt order and variable renaming, model probability calibration, and whether world-information-gain actions differ from query-risk actions. Mark these secondary analyses clearly. If the model only succeeds when given explicit candidate tables, say so. If it fails, retain the failure rather than tuning on the locked set.

## 9. Milestones and stop conditions

| Milestone | Deliverable | Completion condition |
| --- | --- | --- |
| M0: scaffold | Existing two-variable world and protocol | Exact invariant tests pass; no model claim |
| M1: finite SCM core | Generic three-variable factorization and intervention enumeration | Gate A passes |
| M2: policies | Random, greedy, and exact finite-horizon policies | Gate B passes |
| M3: benchmark | Generator, strata, locked split, manifests, analysis specification | Gate C passes and opportunity gap is measurable |
| M4: model adapter | Structured prompts, parser, run ledger, cost limiter | Development smoke test passes |
| M5: evaluation | One frozen open-weight model condition | Gate D passes within authorized budget |
| M6: report | Reproducible tables, figures, and limitations | Gate E passes; raw records support every claim |

Stop or revise **before** model evaluation if observational equivalence, intervention math, split integrity, or the opportunity gap fails. Stop paid work at the hard cap or on repeated infrastructure failures. After a negative locked result, publish the audited negative finding or preregister a genuinely new study version; do not quietly replace the test.

## 10. Resource and security plan

M1-M3 should run on a local CPU. M4-M5 may require hosted inference or a GPU, but the plan assigns no dollar authorization. Before any paid run, record the selected model, measured development throughput, estimated maximum calls and tokens, hard dollar cap, and automatic stop condition. Use an open-weight model first; a second model is optional only after reviewing the first completed study.

The GitHub repository is private. Keep generated episodes, raw responses, provider receipts, secrets, and large artifacts out of Git unless a later release review explicitly selects sanitized outputs. Do not expose credentials in prompts, manifests, logs, or reports. Commits, pushes, and publication are separate actions from implementing the study.
