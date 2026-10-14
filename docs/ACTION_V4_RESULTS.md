# Action-choice benchmark v4: presentation and computation diagnostics

Status: all five locked conditions completed 450/450 test cases. The 235B thinking-model ledger was independently replayed on the GPU host and locally; all five conditions passed the local verifier. Both RunPod pods were deleted after artifact collection.

## Locked question and design

V3's unaided 32B, 72B, and 235B conditions each selected `do(Y=0)` in all 450 cases. V4 asks whether the fixed choice follows the intervention label or the presented action ID, whether public likelihood tables help calculate action values, and whether a 235B thinking checkpoint can choose an informative action from raw candidate SCMs.

The frozen v4 manifest is `data/action-v4-locked.json`, hash `74de5a1614fb6ffa67201c691a18a8830cfacbeb24f92a0072836275f7f86298`. It contains 36 development and 450 test episodes generated with seed 413927. The 300 informative test cases have 100 optimal examples for each semantic action; their randomized presented IDs are nearly balanced at A/B/C = 101/101/98. Another 150 near-tie cases are secondary. The mean informative random risk is 0.129469 nats; exact one-step risk is 0.102272; the opportunity gap is 0.027196.

The score is prior-averaged one-step Bayes risk for the selected action, averaged over candidate worlds and all possible experimental outcomes. Lower is better. An invalid response or `STOP` receives initial no-experiment risk. The model-minus-random intervals use 2,000 bootstrap resamples of independent joint-parameter draws over the 300 informative cases. The [prospective rationale](test-count-rationale-v4.md) registers the conditions and makes the 235B thinking-model difference from random the confirmatory comparison. The 32B intervals and paired contrasts are descriptive diagnostics.

## Verified test results

The four 32B conditions used pinned `Qwen/Qwen2.5-32B-Instruct-AWQ` revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, vLLM 0.10.2, the same H100 SXM, 96-token direct JSON output cap, temperature 0.7, top-p 0.8, top-k 20, and per-prompt seeds. Every condition completed all 450 test calls with zero invalid actions. Both the remote and local independent verifiers replayed each ledger.

| Condition | Mean model risk | Model minus random | 95% interval | Optimal informative actions | Presented ID choices |
| --- | ---: | ---: | ---: | ---: | --- |
| Literal action IDs, raw SCM | 0.126802 | -0.002666 | [-0.006155, +0.000751] | 100/300 | `do(Y=0)` 450/450 |
| Opaque A/B/C IDs, raw SCM | 0.131019 | +0.001551 | [-0.001237, +0.004396] | 102/300 | B 308, C 142, A 0 |
| Opaque IDs, public likelihood table | 0.130185 | +0.000717 | [-0.002222, +0.003753] | 109/300 | B 269, C 164, A 17 |
| Opaque IDs, exact action values | 0.102272 | -0.027196 | [-0.030334, -0.024161] | 300/300 | A 148, B 147, C 155 |
| Opaque IDs, 235B thinking | 0.130978 | +0.001509 | [-0.001350, +0.004370] | 102/300 | A 2, B 305, C 143 |

The 235B condition used pinned `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` revision `f07f63f2bbd7540917118ebdf3812696ef303b03` on four H200 GPUs, vLLM 0.10.2, 2,048 output tokens including at most 1,536 thinking tokens before a constrained final JSON action, temperature 0.6, top-p 0.95, top-k 20, and per-prompt seeds. It completed 450/450 calls with zero invalid actions. Its registered model-minus-random interval crosses zero, so the better-than-random criterion was **not met**. Its mean risk was 0.130978 nats versus 0.129469 for uniform random action choice.

The literal prompt reproduced the fixed semantic `do(Y=0)` choice. Randomizing the intervention mapping changed the semantic choices, but the model heavily preferred B and C as presented IDs. The opaque raw condition did not beat random. Supplying per-world query probabilities and experiment-outcome likelihoods moved the mean risk by only -0.000834 nats relative to opaque raw, with paired interval [-0.002972, +0.000959]. Exact action values produced the exact policy on all 300 informative cases. Those values supply the computational answer, so this is a selection control rather than evidence of unaided causal calculation.

The paired opaque-minus-literal difference was +0.004217 nats, interval [-0.000473, +0.009609]. The 235B thinking-minus-32B opaque-raw difference was -0.0000415 nats, descriptive paired interval [-0.000255, +0.000130]. Both chose an optimal action on 102/300 informative cases, and both heavily favored presented ID B. These results show no detected improvement from the 235B thinking checkpoint under this protocol. Because the checkpoints differ in architecture, training, decoding, and GPU execution, the contrast does not isolate parameter count or thinking as a causal factor. These are descriptive comparisons on one synthetic task with one intervention choice and a fixed response schema. They do not establish a general position bias or causal reasoning limit.

## Audit receipts and compute

| Ignored local ledger | SHA256 of `runs.jsonl` |
| --- | --- |
| `results/v4-test32-literal-raw` | `c091fc7a79df8f37369d9019c1ed3a98d23f527b74d56f243452e4880f61d83a` |
| `results/v4-test32-opaque-raw` | `3ade55380a1ec0ce9d7d40caa5222b29f103062c142fe8f4f83bcf6622b94b1b` |
| `results/v4-test32-opaque-likelihood` | `df938057730c90e90c33c1f91be344c9699783d29aaca90bbc883cd2ee367a2e` |
| `results/v4-test32-opaque-values` | `e5c35f8d0038c17ff0a1ab44601f7e72b4476f3bf2ce9636c813e1ccc9049205` |
| `results/v4-test235-opaque-thinking-batched` | `0fdc41f3077ee45774533c848c6803829bbf7bbfc8f1463e0e2200a36ab647e2` |

The separate verifier rebuilds the prompt, maps each presented ID back to its intervention, independently computes action risks with `BeliefModel` and `one_step_risk`, and checks the saved score. All five complete ledgers passed. The 235B ledger passed on both the GPU host and macOS, including the prompt and response replay. The 61 local unit tests pass. Source bytes matching the v3 and v4 freeze hashes are archived under ignored `artifacts/action-v3-source/` and `artifacts/action-v4-source/`. The post-freeze batched runner is separately archived under ignored `artifacts/action-v4-source-amendment/`, SHA256 `a1d9845b712d9a7ae557730d9a7012f4cc3facbdf5f6664aa783662d34ef0ab0`.

The H100 pod `5xhh27yvlos0nb` was created at $3.99/hour and deleted after all its results were copied and verified locally. The four-H200 pod `fvojku7v5kozfh` was created at $21.16/hour and deleted after the full 235B ledger and run log were copied and independently verified. Follow-up `get_pod` calls returned 404 for both; `list_pods` was empty after the H200 deletion. RunPod's subsequently posted pod billing totals $39.15, including GPU and disk charges for both pods. The approved v4 cap was $75 including storage.

## Larger thinking-model condition

The initial serial development case returned a valid action but required 137.1 seconds for 1,544 output tokens. Serial execution of 450 cases would exceed the approved cap. Before the first 235B test call, a 16-case batched development pilot completed in 146.0 seconds of generation time with 16 valid responses. The [execution amendment](action-v4-batch-amendment.md) records the separate runner digest and the fact that batched scheduling can change sampled text. The complete locked test took 4,620.3 process seconds, including checkpoint loading, under its 6,000-second limit. Batches preserved each prompt's seed and sampling settings, but batching may change sampled text compared with serial execution. No test case, prompt, score, model revision, or registered decision rule was changed after test calls began.
