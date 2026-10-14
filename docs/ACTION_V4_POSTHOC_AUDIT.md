# V4 post hoc label and reasoning audit

Status: exploratory analysis of the completed, frozen v4 test. It was designed after the model results were seen and does not revise the registered v4 decision rule. Reproduce the quantitative results with `PYTHONPATH=. python3 scripts/audit_action_v4.py data/action-v4-locked.json results/v4-test32-opaque-raw results/v4-test235-opaque-thinking-batched --output artifacts/v4-posthoc-audit.json`. The script first replays both complete ledgers with the independent verifier.

## Choice behavior on 300 informative cases

The oracle's best presented label was A/B/C in 101/101/98 cases. The 32B opaque-raw model chose B/C in 212/88 cases and never chose A. The 235B thinking model chose A/B/C in 1/210/89 cases. Their labels agreed on 298/300 informative cases and 447/450 test cases overall. Each selected an optimal action in 102/300 informative cases, compared with 101/300 for a policy that always chooses B.

For the 235B model, B was chosen in 71/101 cases where B was optimal and 139/199 where it was not. Those rates are 70.3% and 69.8%. The corresponding 32B rates were 72/101 (71.3%) and 140/199 (70.4%). This diagnostic shows little apparent sensitivity of the B choice to whether B was optimal. It does not prove the model ignored every numerical input.

On these same cases, always choosing B had mean risk 0.002998 nats below the uniformly random action. The 235B model had risk 0.004507 nats above always-B, with an exploratory paired bootstrap interval [0.001162, 0.008474]. The 32B difference was +0.004549, interval [0.001179, 0.008486]. Because the favored B label was selected for this comparison after inspecting the v4 responses, these intervals are descriptive and are not a registered test against a prespecified baseline. Fresh data are needed to confirm any general label effect.

## Reasoning trace limitation

The 235B ledger saved a final JSON action and a raw thinking trace for every case. Its total output token counts were 1,544 in 435 cases and 1,547 in 15 cases. The runner allowed 1,536 tokens in the first thinking pass, then constrained the final action. In a reproducible sample of eight informative cases selected with Python `Random(413927).sample(range(300), 8)`, the saved traces were still deriving causal factorization or marginals when the final action was emitted. These observations are consistent with the thinking pass frequently reaching its cap. The ledger does not separately store the first-pass finish reason or token count, so it cannot establish exactly how many cases exhausted that cap.

The v4 responses did not require structured intermediate calculations. Their prose therefore cannot be scored reliably for the first incorrect step. A follow-up must request and separately score numerical intermediate outputs, while keeping the held-out action and exact risk values out of the model input.

## Decision

The completed v4 result remains negative under its registered better-than-random rule. Its most useful next question is whether models can calculate and use the public one-step risk in smaller, independently scored steps. A model with a correct calculator should be evaluated as a tool-using system, not as unaided causal reasoning. Another larger-model raw prompt alone would leave the observed label preference and computation bottleneck unresolved.
