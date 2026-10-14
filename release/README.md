# Public evidence bundle

This directory contains the exact locked manifests and model-response ledgers used in the technical paper. The SHA-256 index binds 69 evidence files. Model weights, Hugging Face caches, credentials, and temporary RunPod disks are not included.

The manifests, ledgers, and index are licensed under [CC BY 4.0](DATA_LICENSE.md); archived software retains the repository's [MIT License](../LICENSE). The README and license notices are documentation, not entries in the evidence hash index.

## Status by version

| Version | Complete locked model tests | Main boundary |
| --- | --- | --- |
| v1 | Qwen3-32B end-to-end forecast | Weak realized random-to-oracle action gap |
| v2 | Qwen3-32B and Qwen2.5-72B | Realized-outcome action score |
| v3 | Five action-only conditions | Fixed semantic action choices |
| v4 | Five action-only conditions | Randomized labels, public likelihood/value controls, 235B thinking |
| v5 | Two 32B diagnostics | Three 235B test conditions unrun after budgeted development check |

The v5 development ledgers include both the original 256-token raw-stage pilot and the amended 512-token pilot. The short 235B ledger contains eight development cases only. These are included to document the execution decision; they are not test results.

## Verify

From the repository root, these commands use only Python's standard library and the repository code:

    python3 repro/verify_public_release.py --full

This single command checks every indexed file, replays all complete v2-v5 test ledgers, and audits the v3-v5 information identity. To replay one condition instead, use the commands below:

    PYTHONPATH=. python3 scripts/verify_action_v4.py release/manifests/action-v4-locked.json release/ledgers/v4-test235-opaque-thinking-batched
    PYTHONPATH=. python3 scripts/verify_action_v5.py release/manifests/action-v5-locked.json release/ledgers/v5-test32-raw
    PYTHONPATH=. python3 scripts/verify_action_v5.py release/manifests/action-v5-locked.json release/ledgers/v5-test32-tool
    PYTHONPATH=. python3 repro/check_information_identity.py release/manifests/action-v5-locked.json

The v3 cross-runtime verifier is scripts/verify_action_v3_portable.py. The v2 verifier checks the frozen source hashes; extract frozen-source/action-v2-source.tar.gz into a separate directory, copy the post-lock scripts/verify_action_v2.py into its scripts/ directory, and run it there with the v2 manifest and ledger paths. The v1 result report records its separate verifier and frozen source bundle. Test sets are closed; reproduction means replaying the saved responses and exact scores, not retuning prompts on these cases.

Run repro/build_public_release.py from the repository root to regenerate this allowlisted evidence bundle from the local ignored data/, results/, and artifacts/ directories. That script overwrites public copies only; it does not generate new model responses.
