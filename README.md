# Choosing Experiments for Causal Prediction

**An exact Bayes-risk benchmark of open language models.** This repository asks a narrow question: when several causal worlds fit the same observational data, can a model choose an experiment that improves a specified future intervention prediction? It contains the [five-page paper](paper/main.pdf), [LaTeX source](paper/main.tex), benchmark code, frozen protocols, complete public result ledgers, and CPU-only verification tools. The [research releases](https://github.com/arpjw/causal-interventions/releases) package the paper for download.

The main finding is a bounded negative result. In locked v3 and v4 studies, unaided 32B, 72B, and 235B model conditions did not meet their registered better-than-random experiment-choice rules. A fresh v5 diagnostic found that a 32B agent did better when it could request exact action risks, but its tool use was incomplete. The registered v5 235B thinking-budget comparison was **not run** after a development-only throughput check showed that full coverage could not be budgeted reliably under the final RunPod cap. These synthetic tests do not measure general causal-reasoning ability.

## Start here

| Goal | Where to go |
| --- | --- |
| Understand the mathematical task and conclusions | [Paper PDF](paper/main.pdf), especially Sections 2, 4, and 5 |
| See measured results and limitations by study | [v1](docs/QWEN32_LOCKED_RESULTS.md) · [v2](docs/ACTION_V2_RESULTS.md) · [v3](docs/ACTION_V3_RESULTS.md) · [v4](docs/ACTION_V4_RESULTS.md) · [v5](docs/ACTION_V5_RESULTS.md) |
| Verify saved results on a CPU | [Reproduce the public evidence](#reproduce-the-public-evidence-on-a-cpu) |
| Find frozen cases and raw model responses | [`release/manifests/`](release/manifests/) and [`release/ledgers/`](release/ledgers/) |
| Inspect the design and deviations | [`docs/`](docs/), including versioned count rationales and execution amendments |

## Research question and exact score

An episode supplies a positive observational joint distribution over three binary variables, two or three candidate causal orders, a prior over the worlds, three allowed interventions, and the held-out query `P(Z=1 | do(X=1))`. Each causal order factorizes the **same** observational joint distribution. Its truncated factorization nevertheless gives different intervention predictions. The agent chooses one experiment from `do(X=0)`, `do(Y=0)`, and `do(Y=1)`; it may also choose `STOP` where the protocol allows it.

The exact evaluator enumerates every possible outcome of an action, updates the posterior over candidate worlds, and measures the remaining Bernoulli log-loss risk for the held-out query. If `R_a` is that expected risk and `R_random` is the average over the three allowed experiments, the action-choice contrast is:

```text
model minus random = mean_case(R_chosen - R_random).
```

**Negative values favor the model.** The oracle minimizes risk over the allowed actions and `STOP`; the random-to-oracle gap measures how much room there is to improve. The paper proves that prior query risk minus `R_a` equals the mutual information between the experiment outcome and a future draw from the held-out query, under the stated conditional-independence convention. [`repro/check_information_identity.py`](repro/check_information_identity.py) checks this identity by a separate finite joint-table calculation on all locked v3-v5 test actions.

The studies use different primary scores. V1 and v2 score realized forecasts; v3-v5 use prior-averaged exact action risk. Their numerical effect sizes should **not** be pooled or read as one model-size curve.

## What the locked studies found

| Study | Complete test coverage | Registered or principal result | Important boundary |
| --- | --- | --- | --- |
| v1 | 360 Qwen3-32B episodes | End-to-end forecast loss minus random: **+0.262067 nats**, 95% interval `[+0.163531, +0.380312]` | Realized random-to-oracle gap was only `0.000567` nats; weak action discrimination. |
| v2 | 150 cases per 32B and 72B condition | Action-conditioned forecast loss minus random: **+0.019137** (32B) and **+0.009999** (72B) nats; both intervals include zero | Different model families and decoding; realized-outcome score. |
| v3 | 450 cases per condition, including 300 informative | Unaided 32B, 72B, and 235B each chose `do(Y=0)` on every case; each was optimal on 100/300 informative cases | Exact-risk action score, but semantic action labels were fixed. Two value-assisted controls selected the optimum in 299/300 cases. |
| v4 | 450 cases per condition, including 300 informative | Registered 235B thinking-model minus random: **+0.001509 nats**, 95% interval `[-0.001350, +0.004370]`; optimal on 102/300 | Randomized action labels. The 32B exact-value selection control was optimal on 300/300, but it was supplied the answer values. |
| v5 | Two 32B conditions on 150 fresh informative cases | Raw numerical stages: **+0.015592 nats** vs random; exact-risk calculator agent: **-0.009773 nats** vs random. Their descriptive paired difference was **-0.025365 nats**. | The agent queried A then B in every case, never C, yet chose C 102 times. The registered 235B long-versus-short test is unrun. |

Intervals resample independent benchmark parameter draws while holding the saved model responses fixed. They do not include variation from retraining or repeated decoding. The v5 calculator comparison is descriptive and measures a tool-using system, not unaided numerical causal reasoning. See the linked reports and paper for full condition definitions, intervals, parse failures, pretest amendments, and failure analysis.

## Reproduce the public evidence on a CPU

These commands work on macOS or Linux with **Python 3.10+** and its standard library. They use the saved model responses and exact finite-table calculations; they do **not** download model weights, require a GPU, call RunPod, or produce new model responses.

```sh
git clone https://github.com/arpjw/causal-interventions.git
cd causal-interventions
python3 -m unittest discover -s tests -q
python3 repro/verify_public_release.py
python3 repro/verify_public_release.py --full
```

The first release command checks the SHA-256 digest and byte count of every file in [`release/sha256.json`](release/sha256.json). `--full` additionally extracts the frozen v2 source into a temporary directory, verifies both v2 test ledgers, replays all five v3 and five v4 test conditions, replays both completed v5 test conditions, and checks the query-information identity on the v3-v5 locks. It checks **14 complete v2-v5 model test ledgers** in total. Expect this to take longer than the unit tests on a laptop. It leaves the repository untouched and exits nonzero on a failed check.

The full checker uses [`scripts/verify_action_v3_portable.py`](scripts/verify_action_v3_portable.py) for v3 because floating-point prompt values can differ in their last bits between Python/libm runtimes; nonnumeric prompt content and model responses must still match. The v2 verifier checks hashes against its archived frozen source, so the checker extracts that archive separately. V1's published manifest, ledger, and exact frozen source bundle are hash-checked, and its original independent replay is documented in the [v1 report](docs/QWEN32_LOCKED_RESULTS.md); the current `--full` command does not independently recompute v1's forecast-loss score.

### Inspect or replay one condition

Each `release/ledgers/NAME/` directory contains `metadata.json`, `runs.jsonl`, and `summary.json`. A row in `runs.jsonl` records the public prompt, saved generation or tool transcript, selected action, and exact case scores. Metadata binds the condition and manifest; the summary holds aggregate results. For example:

```sh
PYTHONPATH=. python3 scripts/verify_action_v5.py \
  release/manifests/action-v5-locked.json release/ledgers/v5-test32-raw
PYTHONPATH=. python3 scripts/verify_action_v5.py \
  release/manifests/action-v5-locked.json release/ledgers/v5-test32-tool
PYTHONPATH=. python3 scripts/verify_action_v4.py \
  release/manifests/action-v4-locked.json release/ledgers/v4-test235-opaque-thinking-batched
PYTHONPATH=. python3 repro/check_information_identity.py \
  release/manifests/action-v5-locked.json
```

The `--full` checker runs the analogous commands for every complete v2-v5 test condition. [`release/README.md`](release/README.md) lists the evidence bundle and its boundaries. Development ledgers, including the eight-case v5 235B throughput pilot, are present for provenance but are **not test results**. Test sets are closed: do not tune a new prompt on them and call the resulting score a fresh confirmatory result.

### Rebuild a new local benchmark candidate

The package itself has no required third-party Python dependencies. To explore the implementation without touching the published locks:

```sh
python3 -m causal_interventions generate \
  --config configs/benchmark-v0.json --output data/my-candidate.json \
  --development-per-stratum 4 --test-per-stratum 4
python3 -m causal_interventions validate data/my-candidate.json
python3 -m causal_interventions audit data/my-candidate.json
```

The v0 file is a small exploratory configuration. [`configs/benchmark-factorial-v0.json`](configs/benchmark-factorial-v0.json) and the v2-v5 configurations document later designs. `data/`, `results/`, and `artifacts/` are Git-ignored local workspaces; public copies of the relevant locked artifacts are under `release/`. Manifest generation uses exclusive creation to avoid silently replacing an earlier candidate. The [engineering plan](docs/ENGINEERING_PLAN.md) and [initial protocol](docs/protocol.md) explain the progression from the M1-M3 exact core to the later model studies.

## Re-running inference with a model

The CPU replay above verifies the **published observations**. New inference is a separate experiment: it needs the exact model checkpoint revision, inference stack, decoding settings, prompt/source bytes, and enough GPU memory and time. Model weights are not redistributed here. V2-v5 locks carry source and prompt hashes; [`release/frozen-source/`](release/frozen-source/) preserves version-specific source bytes after the working tree evolved. V4's batch execution amendment and v5's pretest 512-token raw-stage amendment are recorded in [`docs/action-v4-batch-amendment.md`](docs/action-v4-batch-amendment.md) and [`docs/action-v5-execution-amendment.md`](docs/action-v5-execution-amendment.md).

The recorded CUDA environment used vLLM 0.10.2 and the pins in [`requirements-runpod-pilot.txt`](requirements-runpod-pilot.txt). Each result report names its model snapshot, decoding rule, hardware, and coverage. [`scripts/run_action_v3.py`](scripts/run_action_v3.py), [`scripts/run_action_v4.py`](scripts/run_action_v4.py), and [`scripts/run_action_v5.py`](scripts/run_action_v5.py) are the original runners; [`repro/run_action_v5_amended.py`](repro/run_action_v5_amended.py) is the documented pretest execution amendment. Replaying saved results requires none of these inference dependencies. Rerunning a model may produce different responses even with the recorded seed and should be reported separately from the published ledgers.

## Repository map

| Path | Contents |
| --- | --- |
| [`causal_interventions/`](causal_interventions/) | SCM factorization, interventions, beliefs, exact policies, episode generation, prompting, scoring, and adapters |
| [`configs/`](configs/) and [`prompts/`](prompts/) | Versioned study inputs and frozen prompt text |
| [`docs/`](docs/) | Protocols, count rationales, results, audits, and amendments |
| [`paper/`](paper/) | PDF, source, and paper license |
| [`release/`](release/) | Public locked manifests, model-response ledgers, frozen source archives, SHA-256 index, and data license |
| [`repro/`](repro/) | One-command release audit, information-identity audit, release builder, and v5 amended runner |
| [`scripts/`](scripts/) and [`tests/`](tests/) | Study runners, independent verifiers, and unit/integration tests |

## Scope, citation, and licenses

This is a deliberately synthetic three-variable, known-candidate-world benchmark with one query form and one action. Informative cases are selected by a model-free opportunity gate. The studies do not establish real-world causal discovery, general model ability, or an effect of parameter count alone. Negative results and incomplete conditions are retained in the public record.

If you use the result, cite **Arya Somu, “Choosing Experiments for Causal Prediction: An Exact Bayes-Risk Benchmark of Open Language Models,” October 2026**, and link to the [paper](paper/main.pdf) and this repository. Code and frozen software are under the [MIT License](LICENSE). The paper is under [CC BY 4.0](paper/LICENSE.md). Published manifests and ledgers are under [CC BY 4.0](release/DATA_LICENSE.md). Model checkpoints retain their upstream licenses and are not included.
