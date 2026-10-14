# V5 pretest execution amendment

Status: recorded after an 18-case **development-only** Qwen2.5-32B raw-stage run and before any v5 test model response. The locked test manifest, 150-case count, prompts, five conditions, primary long-minus-short risk comparison, practical threshold, and verifier remain unchanged.

The frozen runner allowed 256 generated tokens for the direct 32B stage JSON. The development ledger showed 10/18 incomplete JSON responses that reached exactly 256 tokens. The remaining responses often chose `STOP`. The independent verifier replayed all 18 rows, including the incomplete outputs. This is an output-length fault in the interface, not evidence that a longer cap will improve action choice.

For both 32B stage conditions, use the amended runner at `repro/run_action_v5_amended.py`, SHA-256 `786f4457fad8783f3f685fcbbb688144d7691453c8f43d722ed1e9c110078c66`. It changes only the direct 32B stage generation cap from 256 to 512 tokens and the corresponding metadata field. The 235B thinking and constrained-final caps, calculator-agent protocol, and all scoring code are byte-identical to the frozen runner. The amended runner is outside the frozen source hash list, so the original locked source remains independently checkable. Full source bytes for the frozen and amended runners are preserved locally.

Rerun the development split with the amended runner and verify its ledger before the 32B test. Do not inspect or use v5 test responses to make another protocol change. Report both development outcomes and this deviation alongside any v5 test result. If the model still selects poor actions, retain that result.
