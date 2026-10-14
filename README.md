# Causal Interventions

A controlled study of whether AI agents choose informative experiments and use their results to predict interventions. Locked studies through v4 are complete. A fresh v5 diagnostic evaluated raw numerical stages against an exact-risk calculator agent on 150 cases; its planned 235B confirmatory comparison was stopped before test inference because measured throughput could not fit the project budget.

## Paper and reproducible release

The [technical paper](paper/main.pdf) and its [LaTeX source](paper/main.tex) derive the exact one-step Bayes-risk objective and prove that the expected risk reduction equals the mutual information between an experiment outcome and the held-out query outcome. They report the full v3 and v4 results, the completed v5 32B diagnostic, and the v5 budget stop. The [v5 result report](docs/ACTION_V5_RESULTS.md) gives the detailed receipts and limitations.

The [release index](release/sha256.json) hashes 69 public evidence files: locked manifests, complete test ledgers, development diagnostics, and archived frozen source. Model weights and credentials are not included. Each ledger retains prompts, responses, exact per-case scores, metadata, and a summary. The paper and code are released as a research artifact, without claiming general causal-reasoning performance.

To verify v5 from the repository root, run scripts/verify_action_v5.py with release/manifests/action-v5-locked.json and each of release/ledgers/v5-test32-raw and release/ledgers/v5-test32-tool. Run repro/check_information_identity.py on any locked v3, v4, or v5 manifest to check the information identity independently. For v4, use scripts/verify_action_v4.py. For v3 across Python runtimes, use scripts/verify_action_v3_portable.py. The public results are historical model responses; rerunning inference requires pinned model snapshots and GPU resources.

## Research question

When observational data cannot distinguish two causal worlds, does an agent choose interventions that reduce uncertainty about what will happen under a new intervention?

The two-variable smoke test contains a pair of causal worlds with the same observational joint distribution. The main benchmark uses three binary variables, candidate causal orders that reproduce the same observational joint exactly, and interventions with different predictive value.

## Current implementation

- Exact three-variable SCM factorization and intervention enumeration (M1).
- Bayesian belief updates, random and greedy baselines, and an exact finite-horizon query-risk policy (M2).
- Deterministic development/test episode generation, informative and near-tie strata, rejection accounting, and replay-validated candidate manifests (M3).
- Development episode runner, strict action and forecast parsing, paired baselines, and optional offline MLX, Transformers, and CUDA vLLM adapters.

The tracked [v0 configuration](configs/benchmark-v0.json) is a small budget-one exploratory setting. The [factorial v0 configuration](configs/benchmark-factorial-v0.json) covers candidate counts two, three, and six and budgets one, two, and three. The first factorial test is frozen and complete; see the [32B locked result](docs/QWEN32_LOCKED_RESULTS.md). Local manifests and ledgers under `data/` and `results/` are ignored by Git. See [M1-M3 status](docs/M1_M3_STATUS.md) for implementation and validation history.

The [v2 action-choice configuration](configs/benchmark-action-v2.json) strengthens the informative action gap and makes the exact Bayesian forecast after model actions the primary score. A budgeted-thinking Qwen3-32B-AWQ condition and a direct-JSON Qwen2.5-72B-Instruct-AWQ condition each completed 150 fresh locked test episodes. Neither showed a resolved improvement over random action choice; both submitted forecast scores were worse than random. See the [v2 locked report](docs/ACTION_V2_RESULTS.md) and [frozen count rationale](docs/test-count-rationale-v2.md). The conditions differ in model family and decoding, so the result does not isolate the effect of model size.

The [v3 action-only study](docs/ACTION_V3_RESULTS.md) is complete. It used 300 informative and 150 near-tie locked test episodes, with balanced optimal action labels in each informative stratum. Its primary metric averages exact posterior query risk over possible experiment outcomes. Unaided Qwen2.5 32B, Qwen2.5 72B, and Qwen3 235B MoE all selected `do(Y=0)` in every case and did not beat random under the registered rule. Both Qwen2.5 models nearly matched the exact policy when supplied the public-input action values. The frozen manifest is ignored local data at `data/action-v3-final-locked.json`; its lock hash is `37d3fbd0dd84be40878c244b87f6258dda3358f3af242c18e32702b88909c99d`. See the [prospective rationale](docs/test-count-rationale-v3.md), full ledgers under ignored `results/`, and [report](docs/ACTION_V3_RESULTS.md) for intervals, independent replay, compute, and limits.

The [v4 action-choice study](docs/ACTION_V4_RESULTS.md) is complete. It randomized presented action IDs on a fresh 450-case locked test, compared raw inputs with public likelihoods and exact action values on Qwen2.5-32B, and ran a budgeted Qwen3-235B thinking condition on four H200 GPUs. The 235B condition did not meet the registered better-than-random criterion: model-minus-random risk was +0.001509 nats on 300 informative cases, 95% interval [-0.001350, +0.004370]. Exact action values led the 32B model to select the optimum on all 300 informative cases. All five ledgers passed independent replay; both pods were deleted. The [prospective rationale](docs/test-count-rationale-v4.md), [batch execution amendment](docs/action-v4-batch-amendment.md), and [full report](docs/ACTION_V4_RESULTS.md) state the design, results, spend estimate, and limitations.

The [v5 diagnostic](docs/ACTION_V5_RESULTS.md) used a fresh 150-case lock. The 32B raw numerical-stage condition was worse than random by +0.015592 nats; the exact-risk calculator agent was better by -0.009773 nats. Their descriptive paired difference was -0.025365 nats. The tool agent always queried A then B and usually chose unqueried C, so this is not evidence of complete tool use or unaided calculation. The registered 235B long-versus-short comparison remains untested after a development-only throughput pilot showed the full study would not reliably fit the $30 cap.

See [the engineering plan](docs/ENGINEERING_PLAN.md) for the complete build and validation path, and [the protocol](docs/protocol.md) for the initial study boundaries.

## Local commands

Requires Python 3.10 or newer and no third-party packages:

```sh
python3 -m unittest discover -s tests -v
python3 -m causal_interventions generate --config configs/benchmark-v0.json --output data/my-candidate.json --development-per-stratum 4 --test-per-stratum 4
python3 -m causal_interventions validate data/my-candidate.json
python3 -m causal_interventions audit data/my-candidate.json
```

The `freeze` command requires a written count rationale and records hashes of the source, prompt, and analysis files. Manifest generation uses exclusive file creation and will not overwrite a prior run. `data/`, `results/`, and `artifacts/` are ignored so local runs and provider outputs stay out of Git.

A RunPod A40 pilot used the pinned [Qwen3-32B-AWQ checkpoint](https://huggingface.co/Qwen/Qwen3-32B-AWQ) with vLLM 0.10.2 and JSON constrained decoding. The 360-episode locked test completed with zero parse failures and a +0.262067-nat paired model-minus-random loss difference, where positive is worse. See the [full result and limitations](docs/QWEN32_LOCKED_RESULTS.md), [test count rationale](docs/test-count-rationale-v1.md), and [CUDA environment pins](requirements-runpod-pilot.txt).

The v2 H100 study used the same pinned CUDA environment. Its primary model-minus-random action-choice differences were +0.019137 nats for 32B and +0.009999 nats for 72B; both 95% intervals crossed zero. The pod was deleted after artifact verification. Public copies of the locked manifests and result ledgers are under release/.

For the v2 independent replay, run `python3 scripts/verify_action_v2.py data/action-v2-locked.json results/runpod-action-v2-locked32` and repeat with `results/runpod-action-v2-locked72`. The verifier is a post-lock audit tool; it was not used to select test conditions.

For a v3 action-only development run, use `PYTHONPATH=. python3 scripts/run_action_v3.py --manifest data/action-v3-portable-candidate.json --model-snapshot SNAPSHOT --output-dir results/NAME --split development --condition base --max-wall-seconds 600 --max-episodes 24`. Use the locked manifest and `--split test` for a complete frozen test; it refuses an episode limit on test. Run `PYTHONPATH=. python3 scripts/verify_action_v3.py data/action-v3-final-locked.json results/NAME` to replay a completed ledger on the generating runtime, or `PYTHONPATH=. python3 scripts/verify_action_v3_portable.py ...` across Python/libm runtimes. Local v2 source bytes matching its frozen hashes are preserved under ignored `artifacts/action-v2-source/` because the main working tree now contains v3 development.
