# First study protocol

## Claim to test

An agent that reasons causally should select experiments that distinguish candidate worlds and improve its predictions about held-out interventions. A language model can do well on observational prediction while failing this test.

## Phase 0: exact controls

Start with a pair of binary structural causal models derived from one strictly positive joint distribution over `(X, Y)`. In one world `X` causes `Y`; in the other `Y` causes `X`. Both reproduce the same observational distribution exactly. Interventions can separate their predictions.

The repository starts with one hand-checkable pair. Before model evaluation, extend the generator to multiple parameter settings, including weak and strong causal effects and cases with little information gain. Validate the observational equality and interventional disagreement for each generated pair.

## Phase 1: baselines

Implement random experiment choice, a greedy expected-information-gain policy, and an exact Bayesian policy over the finite candidate-world set. Specify the initial world prior and observation noise. Score forecasts using log loss; compare experiment choices using expected information gain and downstream prediction loss.

## Phase 2: model evaluation

Evaluate at least one open-weight language model with a fixed prompt format. Give all policies the same observations, action set, intervention budget, and outcome samples. Preserve exact prompts and responses. Randomize variable names and presentation order to measure format sensitivity.

## Evidence rules

- Reserve parameter combinations and random seeds before looking at model results.
- Distinguish structural identification from predictive success when the budget cannot identify the world.
- Use paired worlds and paired random draws where possible; report uncertainty across worlds, not just repeated prompts from one world.
- Record model version, decoding settings, token or call costs, parse failures, and any excluded trials.
- A simulator check or baseline implementation does not establish an agent capability.

## Scope and budget gate

The simulator and exact baselines can run locally without paid services. Decide and record a hard cost cap before any hosted inference or GPU training. Do not run paid experiments as part of repository setup.
