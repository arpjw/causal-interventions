# V5 final diagnostic and budget stop

Status: the two 32B test conditions completed all 150 fresh informative cases and passed independent replay on the RunPod host and locally. The three registered 235B test conditions were **not run**. The v5 confirmatory long-versus-short thinking criterion therefore has no result. Both temporary RunPod pods were deleted, and the final pod list was empty.

## Frozen design and pretest amendment

The frozen test manifest is data/action-v5-locked.json, hash f7da700fb34d1f938e75723cc0fc82ca407e79ed035f6c306537003830a81301. It has 18 development and 150 test cases, all informative, with no joint-parameter overlap with v4. The mean exact random-to-oracle opportunity is 0.027566 nats. The five conditions and the single registered confirmatory 235B longer-minus-shorter criterion are in docs/test-count-rationale-v5.md.

The first 18-case 32B raw-stage development pilot hit a 256-token JSON output cap in 10 cases. Before any v5 test call, docs/action-v5-execution-amendment.md recorded a change to a 512-token cap for that direct JSON call only. The amended runner digest is 786f4457fad8783f3f685fcbbb688144d7691453c8f43d722ed1e9c110078c66. The repeat 18-case development run had zero invalid final actions. The test manifest, prompt, model revision, score, and registered decision rule did not change. The original and amended development ledgers are retained.

## Complete 32B test results

Both conditions used pinned Qwen/Qwen2.5-32B-Instruct-AWQ revision 5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c, vLLM 0.10.2, one Secure Cloud H100, and the same 150 test cases. The raw-stage condition submitted per-world query probabilities, one experiment likelihood, a posterior, all three action risks, and a final action. The calculator agent could request exact public-input risk for at most three named actions and then submit a final action.

| Condition | Mean risk, nats | Minus random | 95% case-bootstrap interval | Optimal / 150 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32B raw numerical stages | 0.141043 | +0.015592 | [+0.012604, +0.018433] | 7 | 0 |
| 32B exact-risk calculator agent | 0.115679 | -0.009773 | [-0.015194, -0.004600] | 76 | 0 |

The uniform-random mean risk was 0.125452 nats and exact-oracle risk was 0.097885. Calculator-minus-raw paired risk was **-0.025365 nats**, with descriptive paired 95% interval **[-0.032520, -0.018185]** over 150 independent parameter draws. This comparison was specified as descriptive, not the v5 confirmatory criterion. It measures a tool-using system against a raw numerical-stage model, not unaided causal reasoning. The calculator condition closed about 35.5% of the random-to-oracle opportunity gap; the raw stage condition was worse than random.

The raw-stage model chose STOP in 134/150 cases. Its stage mean absolute errors were 0.411 for candidate-world query probabilities, 0.287 for a prespecified experiment likelihood, 0.213 for the corresponding posterior, and 0.538 nats for action risks among 148 numerically valid risk objects. These errors can propagate across stages and cannot by themselves identify one first failure.

The calculator agent requested A then B in **all 150 cases**, never requested C, and nevertheless selected C in 102. It selected B in 43 and A in 5. An exact policy choosing the lower of the two requested A/B values would have mean risk 0.112236 nats, below the agent's 0.115679. This post hoc computation shows substantial remaining tool-use inefficiency. Neither the calculator result nor its interval licenses a claim that the model can compute the exact risks itself.

The two runs.jsonl SHA-256 digests are f401cdd07855c38bbb0d13496eca939e9824ab500db8b255abac41ac86f29123 (raw) and 7006bfffd1c2741d7625cf5b0ab2c4ed587c520b3e608ec4e0b2952111dd1513 (calculator). The complete rows, prompts, responses, tool transcripts, per-case risks, metadata, and summaries are in the ignored local results/v5-test32-raw/ and results/v5-test32-tool/, with public copies under release/ledgers/. Both passed scripts/verify_action_v5.py on the GPU host and on macOS.

## 235B development throughput and stop

Pinned Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 revision f07f63f2bbd7540917118ebdf3812696ef303b03 loaded on two B200 GPUs. An eight-case **development-only** short-budget batch took 199.39 seconds of generation after model loading and produced eight valid final actions. Its eight saved rows were replayed locally. This pilot is not a test result and cannot estimate the registered 150-case contrast. The observed rate, the 3,584-token long condition, and the 150-case coverage requirement did not allow a reliable plan to finish the remaining 235B conditions within the new $30 total RunPod cap. No v5 235B test inference was launched. The B200 pod was deleted after copying the development ledger.

The H100 pod 1xacel7c70zhcq and B200 pod 0by3l8a2dd0l8z were both deleted; a final pod list returned empty. RunPod's billing feed had not posted their final charges at cleanup. From catalog GPU rates and observed pod lifetimes, combined GPU cost was approximately $3.75 plus disk, well below the $30 cap; this is an estimate, not a final invoice.

## Interpretation

The completed 32B diagnostic demonstrates that this particular raw numerical-stage interface performed poorly on a fresh exact-risk benchmark while exact tool values helped a bounded agent. The tool agent's fixed A/B request sequence and frequent choice of unqueried C limit the mechanism claim. The registered v5 235B primary comparison remains unanswered. V3 and v4 already supply complete, locked negative results for unaided larger-model action choice; this incomplete v5 study should not be presented as a completed five-condition result.
