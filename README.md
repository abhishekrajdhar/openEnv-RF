---
title: Support Queue OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - reinforcement-learning
---

# Support Queue OpenEnv

`support_queue_openenv` is a real-world agent environment for customer support triage and resolution in an e-commerce operations setting. The agent must inspect records, look up policy, route the case, tag it correctly, draft a customer-facing response, and submit a final resolution. This is the kind of multi-step workflow support teams actually do every day, which makes it useful for both RL training and evals.

## Why this environment

Most agent environments are either games or generic tool-use sandboxes. This one models a practical back-office workflow with measurable correctness, partial progress, and realistic failure modes:

- Agents need to gather evidence before acting.
- Small mistakes matter: wrong route, wrong priority, wrong refund amount, missing VIP handling.
- Reward is shaped across the trajectory rather than only at the terminal step.
- Grading is deterministic and reproducible.

## Environment API

The environment implements the standard `reset()` / `step()` / `state()` API.

- `reset(task_id: str | None = None) -> CustomerSupportObservation`
- `step(action: CustomerSupportAction) -> StepResult[CustomerSupportObservation]`
- `state() -> CustomerSupportState`

`step()` returns:

- `observation`: typed `CustomerSupportObservation`
- `reward`: float trajectory delta
- `done`: episode completion flag
- `info`: structured diagnostics, including grader state

The project also defines a typed `CustomerSupportReward` model, exposed on each observation as `reward_details`, so agents can learn from richer reward decomposition while still complying with the standard scalar reward return.

## Action Space

`CustomerSupportAction` supports these operations:

- `search_policy(argument)`
- `open_order(argument)`
- `open_account(argument)`
- `open_log(argument)`
- `set_priority(argument)`
- `route_ticket(argument)`
- `add_tag(argument)`
- `draft_reply(message)`
- `submit_resolution(resolution)`

`submit_resolution` uses a typed `ResolutionPayload` with:

- `resolution_code`
- `refund_amount`
- `goodwill_credit`
- `shipping_refund`
- `message`

## Observation Space

Each `CustomerSupportObservation` includes:

- task instructions and structured task summary
- currently visible artifacts (order, policy, account, logs)
- selected tags, route, priority, and reply draft
- action history
- remaining steps
- typed reward breakdown in `reward_details`

## State Space

`CustomerSupportState` includes the current episode id, step count, visible artifacts, route/priority/tags, cumulative reward, progress score, last submitted resolution, and evaluation snapshot.

## Tasks

Three deterministic tasks are included, with escalating difficulty:

1. `delayed_shipping_refund` (`easy`)
   Refund the shipping fee for a shipment that is more than five days late, route to logistics, and respond appropriately.
2. `defective_return_window` (`medium`)
   Handle a defective appliance within the return window, add multiple tags, route to returns, and approve a full refund.
3. `subscription_cancellation_dispute` (`hard`)
   Resolve a VIP customer’s billing dispute after a failed cancellation automation event, requiring log lookup, urgent billing routing, refund, and goodwill credit.

## Reward Design

Reward is shaped throughout the trajectory:

- positive signal for discovering required evidence
- positive signal for correct route, priority, tags, and reply content
- final score boost on successful `submit_resolution`
- penalties for repeated actions, invalid actions, out-of-order API use, hallucinated lookups, post-terminal actions, and running out of steps

The deterministic grader produces a final score in `[0.0, 1.0]`. Per-step scalar rewards are clipped into `[0.0, 1.0]` while still preserving penalty information inside `reward_details`.

## Project Layout

```text
support_queue_env/
  __init__.py
  compat.py
  models.py
  tasks.py
  graders.py
  client.py
  server/
    app.py
    support_queue_environment.py
scripts/
  run_baseline.py
inference.py
server/
  app.py
  Dockerfile
tests/
  test_environment.py
openenv.yaml
Dockerfile
```

## Setup

Install locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

If you want to validate against the upstream OpenEnv package as well:

```bash
pip install -e .[dev,openenv]
openenv validate
```

Run the server locally:

```bash
uvicorn support_queue_env.server.app:app --host 0.0.0.0 --port 7860
```

Run tests:

```bash
pytest
```

## Docker

Build and run:

```bash
docker build -t support-queue-openenv .
docker run --rm -p 7860:7860 support-queue-openenv
```

Health check:

```bash
curl http://localhost:7860/health
```

Runtime metadata:

```bash
curl http://localhost:7860/metadata
```

## Hugging Face Spaces

This repo is prepared for a Docker Space. The front matter at the top of this `README.md` sets `sdk: docker` and includes the `openenv` tag. Pushing the repository to a new HF Space is enough for deployment once the Space is created.

Recommended Space settings:

- SDK: Docker
- Port: `7860`
- Tag: `openenv`

## Inference Script

The required inference runner is [inference.py](/Users/abhishekrajdhardubey/Documents/Projects/scaler/inference.py). It uses the OpenAI client and reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY`

It logs exactly:

```text
[START] Task=<task_id>
[STEP] reward=<float> done=<bool>
[END] total_reward=<float>
```

```bash
export OPENAI_API_KEY=...
export MODEL_NAME=gpt-4.1-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

The script runs all three tasks in a fixed order with deterministic prompts, `temperature=0`, and `seed=0`.

## Reproducibility

Reproducibility comes from:

- deterministic task data
- deterministic graders
- fixed task order in the inference script
- `temperature=0` for model calls

## Baseline Scores

The inference script is included and deterministic, but I did not execute it in this workspace because no valid `OPENAI_API_KEY` was available during implementation. Once credentials are present, `python inference.py` will emit reproducible per-task reward traces.
