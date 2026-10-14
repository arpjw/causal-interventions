# First locked 32B GPU evaluation

Status: completed locked evaluation on October 13, 2026. This is one synthetic benchmark condition, not a claim about general causal reasoning.

## Registered condition

- Model: `Qwen/Qwen3-32B-AWQ`, Hugging Face revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`.
- Inference: vLLM 0.10.2, Transformers 4.57.3, JSON-object constrained decoding, temperature 0.7, top-p 0.8, top-k 20, seed from the prompt SHA-256. No retry for an invalid answer.
- Hardware: one RunPod Secure Cloud NVIDIA A40 48 GB GPU in EU-SE-1.
- Lock: `data/qwen32-locked-v1.json`, manifest hash `9f4358e503356189137b32146f5d59190b5563aefa0a6217299f0e5064814798`. It contains 360 test episodes, with 20 independent parameter draws in each of 18 factorial strata. The count and analysis were fixed before the test calls. The candidate also has 72 separate development episodes.
- Primary outcome: paired model-minus-random mean excess Bernoulli log loss in nats; negative would favor the model. Cluster bootstrap resamples independent joint-parameter draws.

## Result

| Policy | Mean excess log loss, nats |
| --- | ---: |
| Qwen3-32B-AWQ | 0.282848 |
| Random | 0.020781 |
| Exact finite-horizon oracle | 0.020214 |

The paired model-minus-random difference was **+0.262067 nats**, with a registered 95% cluster bootstrap interval of **[+0.163531, +0.380312]**. The model performed worse than random on this test. The realized random-minus-oracle gap was only +0.000567 nats, so a gap-closure ratio would be unstable and is not used to describe performance.

All 360 registered test episodes and 1,080 model calls completed. There were zero action/forecast parse failures, zero missing or malformed rows, and zero forecast clipping events. The run generated 958,181 input tokens and 17,049 output tokens. Model generation plus setup and per-episode CPU work took 1,102.8 seconds in the runner. The zero parse-failure rate establishes interface compliance; it does not establish useful causal predictions.

The 18-episode development pilot with JSON constrained decoding also had zero parse failures and a +0.1495-nat model-minus-random difference. The earlier unconstrained development pass had 37 parse failures in 18 episodes. These development passes informed the output format before the fresh test manifest was frozen. No test result informed a prompt or decoder change.

## Audit and artifacts

The local ignored files under `data/` and `results/` contain the locked manifest, raw prompt and response ledger, metadata, and summary. An independent verifier checked every test episode ID and hidden-world identity against the locked manifest, recomputed all query intervention probabilities directly from the joint table and causal order, and recalculated all three policy losses. It matched the runner's +0.262067-nat result and zero parse failures.

| Local artifact | SHA-256 |
| --- | --- |
| `data/qwen32-locked-v1.json` | `c47df01cf72555a3a67b5efa6328c63480ad9163e0d05605c7cef7e641cce39f` |
| `results/runpod-qwen32-awq-locked-v1/runs.jsonl` | `064f440ad199c99db766ecfec1444e4c8ac6b5ff5d65890ef4bbc132ba5077dc` |
| `results/runpod-qwen32-awq-locked-v1/summary.json` | `f466e89fb61addafb940ff88236771eb0167b1ae33bb2b6f80b960178d2eedbc` |
| `results/runpod-qwen32-locked-sources-v1.tar.gz` | `aac9714d8ad71fd4fd01f4671d6451d272c53fd816651f62765075b787dea12a` |

The frozen source list accidentally included 20 macOS AppleDouble sidecar files created during transfer. They did not affect execution. The exact 44-file source bundle was archived, and every file in that archive matches the locked source hashes. The source-hash collector was corrected **after** this run to exclude sidecars in future freezes. The current checkout therefore differs from the frozen source bundle; use the archive to reproduce this specific lock.

RunPod pod `thqiu158hfryxj` was created at 21:09:35 UTC, charged at a displayed $0.59 per GPU hour, and deleted after artifact collection. A provider lookup returned 404 after deletion. Based on roughly 49 minutes of lifetime, compute is estimated near $0.48, plus storage and any provider billing adjustments. This is an estimate, not a final invoice, and is below the approved $10 cap.

## Interpretation and next experiment

This fixed numeric prompt and quantized 32B model produced valid structured replies but substantially worse intervention forecasts than the random policy. A small number of large forecast errors contributed to the excess loss. The study does not isolate whether the failure came from causal inference, numerical calculation, calibration, the prompt interface, AWQ quantization, or the chosen model. The factorial benchmark also does not separately balance weak and strong causal effects. Any revised prompt, calibration method, full-precision model, or larger model needs a new development cycle and a fresh locked test manifest; this test set should not become a tuning set.
