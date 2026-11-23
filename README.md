# Weight-space understanding of RL policies

This project investigates whether access to a frozen RL policy's numerical parameters improves an LLM's predictions about that policy's vulnerabilities. Phase 1 provides local classic-control environments, training, and checkpoint handling. The Phase 1 smoke runs check the infrastructure.

## Reproduce Phase 1

With uv installed, run these commands from the repository root. They use Python 3.12.12, the provided dependency lock, and a separate environment so an existing IDE environment can still be used.

```sh
export UV_PROJECT_ENVIRONMENT=.venv-uv
uv sync --locked --python 3.12.12
uv run --locked --python 3.12.12 pytest -q
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.training \
  --seed 42 --timesteps 4096 --output-root runs/phase1-smoke
```

In PyCharm, select the existing `.venv-uv/bin/python` interpreter for this project. If uv is installed in `~/.local/bin` and your shell cannot find it, use `~/.local/bin/uv` in the commands above.

The training command prints its run directory. Use `--interventions path/to/settings.json` with a configuration such as `{"mask_probability": 0.1}`. Omitted settings use the nominal defaults.

## Environment and policy choices

Choose an environment with `--env`, a training algorithm with `--algorithm`, and an MLP architecture with `--policy`. All combinations use the same training entry point.

| Algorithm | Training and action selection |
| --- | --- |
| `ppo` (default) | On-policy actor-critic with a clipped policy objective. |
| `a2c` | On-policy advantage actor-critic. |
| `dqn` | Off-policy Q-learning with replay and a target network. Deterministic prediction selects the action with the highest online Q-value. |

| Environment | Observations | Actions |
| --- | ---: | ---: |
| `CartPole-v1` (default) | 4 | 2 |
| `MountainCar-v0` | 2 | 3 |
| `Acrobot-v1` | 6 | 3 |

| Policy | Hidden layers per network | Activation |
| --- | --- | --- |
| `tanh` (default) | 8 | Tanh |
| `relu` | 8 | ReLU |
| `deep-tanh` | 8, 8 | Tanh |

For example:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.training \
  --env Acrobot-v1 --algorithm dqn --policy deep-tanh --seed 42 --timesteps 128
```

All three environments support observation masking and action perturbations. CartPole also supports position and velocity noise, gravity changes, and pole-length changes. Match the environment, algorithm, and architecture when assigning incorrect-weight controls, and establish nominal policy competence before comparing vulnerabilities. In particular, nominal returns in MountainCar and Acrobot can be negative, so evaluation will need an explicitly defined degradation scale for those tasks.

## Environment settings

| JSON field | Default | Effect |
| --- | ---: | --- |
| `position_noise_std` | 0 | Add zero-mean Gaussian noise to cart position, in meters. |
| `velocity_noise_std` | 0 | Add zero-mean Gaussian noise to cart velocity, in meters per second. |
| `mask_probability` | 0 | Independently zero each observation coordinate with this probability. |
| `gravity_scale` | 1 | Multiply the simulator's nominal gravity. |
| `pole_length_scale` | 1 | Multiply the simulator's pole half-length and update the mass-length term. Pole mass is fixed. |
| `action_flip_probability` | 0 | Replace the requested action with a uniformly sampled different action with this probability. |

Observation corruption runs on reset and every step, with masking applied after noise. The observation vector remains `[cart position, cart velocity, pole angle, pole angular velocity]`. Cart position has unbounded declared observation bounds in every variant to accommodate Gaussian noise and keep spaces compatible across checkpoints. The action space and CartPole time limit stay the same. An explicit reset seed restarts the initial-state generator and each intervention stream. An unseeded reset continues those streams. The physical effects of gravity and length can interact when both are enabled.

## Training records

Training uses CPU and one PyTorch thread. PPO and A2C use separate actor and critic networks and collect 128 steps per rollout. DQN uses online and target Q-networks, starts gradient updates after 100 environment steps, and collects four steps per training iteration. Requested training counts can round up to the next collection boundary. Metadata records both counts, the algorithm, and the network architecture. The default CartPole PPO policy's action path has 58 parameters, including biases. Checkpoints support frozen-policy loading. DQN replay buffers are outside the saved archive.

Each run has its own checkpoint ID and directory. A completed run contains `policy.zip` with the native SB3 parameters, `metadata.json` with configuration, runtime versions, the checkpoint digest and training summary, and `training_episodes.jsonl` with completed episode returns and lengths. Training exceptions record the failure in the same directory. A process killed abruptly can leave a record marked `running`. Use `wsprl.training.load_checkpoint` on trusted local run directories. Training returns are collected under the configured intervention and a changing, stochastic policy. Use a separate evaluation protocol to establish frozen-policy performance.

Phase 1 was implemented and tested with AI coding assistance. Seeded behavior was verified on one Apple Silicon runtime. Training and trajectory equivalence across platforms or dependency versions remains unverified.

## Batch training

Preview the default grid before starting it:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.batch --dry-run
```

Run it by replacing `--dry-run` with `--output-root runs/pilot-grid`. By default, the grid trains PPO, A2C, and DQN using the same `tanh` architecture, with independent seeds 0, 1, and 2 in every planned group. Use `--seeds 10 11 12` to set the replicates, repeat `--policy` to include additional architectures, or set `--timesteps` to override the exploratory default of 4,096 steps.

| Condition | Severity | Environments |
| --- | ---: | --- |
| Nominal | Defaults | All three |
| Observation masking | Probability 0.1 | All three |
| Action perturbation | Probability 0.1 | All three |
| Cart position noise | Standard deviation 0.1 m | CartPole |
| Cart velocity noise | Standard deviation 0.1 m/s | CartPole |
| Gravity | Scale 1.25 | CartPole |
| Pole length | Scale 1.25 | CartPole |

The defaults produce 117 training runs. To choose other severities or include combined interventions, pass `--conditions conditions.json`. This replaces the defaults with an array of named conditions. Omit `environments` to include all three, or list the applicable environments explicitly. For example:

```json
[
  {"name": "nominal"},
  {"name": "mask-20", "interventions": {"mask_probability": 0.2}},
  {
    "name": "long-pole",
    "environments": ["CartPole-v1"],
    "interventions": {"pole_length_scale": 1.5}
  }
]
```

The output root contains `manifest.json`, which lists each job and its full training configuration. Checkpoints and logs live under `<environment>/<algorithm>/<architecture>/<condition>/seed-<seed>/<checkpoint-id>/`. Rerun with the same settings and output root to reuse verified completed checkpoints and start fresh attempts for unfinished, failed, or corrupt runs. Earlier attempts remain in their directories. A changed plan requires a new output root. Training episode returns are descriptive. Average frozen-policy performance after evaluating each seed independently under the same protocol.

## Phase 2 exploratory evaluation

Evaluate the original six-policy pilot grid with:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.evaluation \
  --batch-root runs/cartpole-pilot-01 --output runs/cartpole-pilot-01-eval
```

The evaluator loads each verified checkpoint and chooses actions deterministically. For each policy, it evaluates nominal performance and two fixed exploratory severity levels per applicable intervention. CartPole levels are position noise 0.05/0.20 m, velocity noise 0.05/0.20 m/s, masking probabilities 0.10/0.30, gravity scales 1.25/1.50, pole-length scales 0.75/0.50, and action perturbation probabilities 0.05/0.15. The two three-action environments use the same masking and action perturbation levels. Every case uses the same 24 pilot evaluation reset seeds per policy, distinct from training and screening seeds.

Absolute degradation is nominal mean return minus perturbed mean return. Normalized degradation divides by the absolute nominal mean and is blank when that mean is zero. Negative degradation means the change improved returns. By default, a return strictly below the nominal mean minus 25% of its magnitude counts as a failure. Change that threshold with `--failure-fraction` and use a new output directory. A paired bootstrap over reset seeds gives a 95% interval for each policy's absolute degradation under a given intervention. Those intervals describe rollout uncertainty within a policy. Comparisons across training conditions use training seeds as replicates.

`protocol.json` records the design and checkpoint identifiers, `raw/episodes.csv` holds the episode records, and `processed/vulnerability.csv` holds per-policy summaries. This command labels its outputs as exploratory pilot data. A held-out study will use a separate cohort and protocol fixed before its evaluation.

## Phase 3 weight export

Export a trusted run directory with:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.serialization \
  --checkpoint path/to/run/checkpoint-id --output runs/local-export
```

The export contains `weights.json` and two local prompt payloads, one with numerical parameters and one with the same task description and architecture. Each named float32 tensor is stored with its shape and all values in row-major order. PPO and A2C export the actor through the categorical logits, then select the action with the highest float32 categorical probability. DQN exports the online Q-network and selects the action with the highest Q-value. The reconstructable path includes float32 casting, vector flattening, hidden activations, and the final linear head. The critic, DQN target network, optimizer, training metadata, and evaluation results stay outside the prompt payloads.

The six CartPole pilot exports each contain 58 parameters. With the `o200k_base` encoding and tiktoken 0.12.0, the message content measured 1,225–1,232 tokens with weights and 645 tokens without weights. These local counts cover the message content. API framing, any response schema, and completion tokens will need a separate estimate once the request format and model are selected.

## Phase 4 LLM harness

Run a local preview with the selected model identifier and its current published prices in USD per million tokens:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.llm_runner \
  --batch-root runs/cartpole-pilot-01 --model MODEL_ID \
  --input-price INPUT_USD_PER_MILLION --output-price OUTPUT_USD_PER_MILLION --dry-run
```

The default preview plans 36 requests for this cohort, covering the four input conditions on six individual checkpoints and three anonymous pairs. Use `--replicates` for independent requests and `--order-seed` for reproducible randomization. Wrong-weight controls use a one-to-one reassignment within the same environment, algorithm, and architecture. The paired wrong-weight condition swaps the two checkpoints. Each replicate draws disjoint pairs within a group, leaving one unpaired checkpoint in groups of odd size. The default paired setting is 30% masking. Use `--paired-intervention` and `--paired-severity` to choose another setting from the evaluation matrix. Match `--failure-fraction` to the evaluation protocol. The no-weight control is naturally shorter, and its tokens are recorded without padding.

Only after approval, replace `--dry-run` with `--execute --ledger runs/llm-ledger.sqlite --experiment-id NAME --request-limit N`. The runner retrieves the configured model before submitting completions and uses the official SDK with automatic retries disabled. Each completion request contains one user message, a structured response schema, and an output-token limit. It supplies neither tools nor conversation history.

Put your API key in a local `.env` file as `OPENAI_API_KEY=your_key_here`. Git ignores that file. Load it into the shell that will run the harness:

```sh
set -a
source ./.env
set +a
```

Reuse the same ledger across experiment names to track cumulative project spending against its configured budget, which defaults to **$8**. Use the same experiment name and configuration to resume. Completed valid and invalid responses are reused. Distinct trial replicates retain distinct cache entries even if their request bodies are identical. The ledger records private assignments, request bodies, response bodies, validations, usage, and estimated and usage-derived charges without authorization headers. Treat it as research data since it links anonymized API requests to checkpoints.

The preview estimates input from serialized request bytes plus a framing allowance and reserves the full output-token limit. Reported input usage is charged at the configured full input rate for local accounting. An incomplete response, refusal, or failed validation is retained as an invalid trial. If billing is uncertain or reported usage exceeds the estimate, execution stops. Unresolved reservations block further calls from that ledger. The ledger budget is a software guard and account billing remains authoritative. For paired responses, ties are recorded explicitly. Accuracy scoring and the treatment of invalid trials in the final analysis require the evaluation protocol to be fixed before the held-out run.

## Pre-Phase-5 audit

Reproduce the independent reset-seed replay and local audit using the Phase 2 cohort:

```sh
export PYTHONPATH=src
uv run --locked --python 3.12.12 python -c 'from pathlib import Path; from wsprl.evaluation import evaluate_pilot; evaluate_pilot(Path("runs/cartpole-pilot-01"), Path("runs/cartpole-pilot-01-audit-replay"), seeds=tuple(range(21000, 21048)))'
uv run --locked --python 3.12.12 python -m wsprl.audit \
  --batch runs/cartpole-pilot-01 --original runs/cartpole-pilot-01-eval \
  --fresh runs/cartpole-pilot-01-audit-replay --output runs/cartpole-pilot-01-audit-v2
```

The diversity audit centers each policy's degradation profile by its mean over intervention settings, then compares pairs of policies. A paired bootstrap resamples entire reset-seed blocks across policies and settings. The 95% band is simultaneous across all policy-pair and setting contrasts within each bank. The original and fresh banks are analyzed separately. The audit reports absolute and normalized contrasts, with a sensitivity analysis using the four checkpoints that passed the earlier nominal screen. The two banks reuse the same trained policies, so these intervals quantify evaluation noise within this pilot cohort.

The audit captures exact SDK requests through an in-process mock transport and saves them in `mock-request-bodies.jsonl`. The accompanying mock ledger contains fabricated responses and usage figures solely for the transport audit. The `weights/` directory contains verified version-2 exports. Version 1 serialized the correct output tensors but described categorical action selection as raw-logit argmax. Version 2 documents SB3's float32 categorical probability argmax, which can select a different action near ties. DQN uses Q-value argmax.

### Complete PPO episode equivalence

Run the behavioral audit against all completed PPO checkpoints in a run directory:

```sh
PYTHONPATH=src uv run --locked --python 3.12.12 python -m wsprl.behavioral_audit \
  --runs runs --output runs/ppo-behavioral-audit
```

The audit reads each exported JSON file back from disk and checks the reconstructed actor's float32 bit patterns against the checkpoint. It runs the original and reconstructed policy in separate environments with the same five reset seeds (42000–42004) per setting. The existing environment wrapper derives its intervention random streams from those reset seeds. The audit requires exact agreement of selected actions, observations, rewards, episode end flags, and returns because both runs use the same CPU runtime and simulator operations. It separately compares logits and full categorical probabilities on every recorded observation and on CartPole decision-boundary probes using the existing tolerance of `atol=1e-7, rtol=1e-6`.

The report `equivalence.json` includes per-checkpoint errors and per-episode outcomes. These seeds are for checking implementation equivalence. Each deterministically executed policy also has its full stochastic action distribution compared on the sampled observations.
