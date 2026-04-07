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

Support Queue OpenEnv is a production-style environment for training and evaluating agents on customer support triage and resolution. Instead of solving a toy game, the agent operates like an internal support operations assistant for an e-commerce company: it reviews the customer issue, retrieves relevant artifacts, applies policy, sets routing and priority, drafts a customer reply, and submits a final resolution.

This environment is designed for OpenEnv-style agent evaluation, reproducible offline benchmarking, and containerized deployment on Hugging Face Spaces.

## Overview

The environment simulates a realistic support workflow with deterministic data and deterministic grading. Each episode is a structured ticket-handling task where the agent must:

- inspect records and policies before acting
- route the case to the correct team
- assign the correct operational priority
- add relevant structured tags
- write an appropriate customer-facing response
- submit a final resolution with the correct financial outcome

This setup creates a useful benchmark for practical agent skills such as evidence gathering, policy compliance, structured decision-making, and long-horizon task completion.

## Why This Environment Matters

Many agent benchmarks over-index on games, synthetic tool use, or loosely specified tasks. This project targets a real operational domain with clear utility:

- customer support is a common, high-value enterprise workflow
- correctness depends on both retrieval and decision quality
- errors are meaningful: wrong route, wrong refund, missed VIP handling, or unsupported escalation
- partial progress can be measured throughout the trajectory, not only at the end

The environment is therefore useful both for reinforcement learning research and for agent evaluation in real-world automation settings.

## OpenEnv API

The environment implements the standard API:

- `reset(task_id: str | None = None) -> CustomerSupportObservation`
- `step(action: CustomerSupportAction) -> StepResult[CustomerSupportObservation]`
- `state() -> CustomerSupportState`

`reset()` initializes a clean episode and returns the initial observation.

`step()` returns:

- `observation`: the next typed observation
- `reward`: scalar reward for the latest transition
- `done`: whether the episode has terminated
- `info`: deterministic evaluation details and diagnostics

`state()` exposes the full internal environment state for debugging, testing, and inspection.

## Typed Models

The environment defines typed Pydantic models for:

- `CustomerSupportAction`
- `CustomerSupportObservation`
- `CustomerSupportReward`
- `CustomerSupportState`

These models make the environment explicit, inspectable, and easier to integrate with OpenEnv tooling.

## Action Space

The agent may choose from the following structured actions:

- `search_policy(argument)`
- `open_order(argument)`
- `open_account(argument)`
- `open_log(argument)`
- `set_priority(argument)`
- `route_ticket(argument)`
- `add_tag(argument)`
- `draft_reply(message)`
- `submit_resolution(resolution)`

Final submission uses the typed `ResolutionPayload`:

- `resolution_code`
- `refund_amount`
- `goodwill_credit`
- `shipping_refund`
- `message`

The environment validates actions and penalizes unsupported or low-quality behavior such as invalid routes, invalid priorities, empty tags, and hallucinated lookups.

## Observation Space

Each `CustomerSupportObservation` includes:

- high-level task instructions
- a structured task summary
- visible artifacts such as order records, policy entries, account data, and logs
- selected tags, route, priority, and reply draft
- action history
- remaining step budget
- a typed reward decomposition in `reward_details`

This allows agents to reason from explicit state rather than brittle free-form text alone.

## State Space

`CustomerSupportState` tracks the full environment state, including:

- episode id
- step count
- current task id
- visible artifacts
- accumulated tags
- priority and routing decisions
- current reply draft
- last submitted resolution
- cumulative reward
- progress score
- evaluation snapshot
- hidden context used for deterministic bookkeeping

## Tasks

The environment includes three deterministic tasks with increasing difficulty.

### 1. `delayed_shipping_refund` — Easy

The agent handles a delayed shipment that has exceeded the promised delivery window. The correct behavior is to inspect the order and policy, route to logistics, add the delay tag, refund the shipping fee, and explain the action clearly to the customer.

### 2. `defective_return_window` — Medium

The agent handles a defective appliance reported within the return window. This task requires the correct return policy lookup, high-priority routing to returns, multiple tags, and a full refund path.

### 3. `subscription_cancellation_dispute` — Hard

The agent handles a VIP billing dispute caused by a failed cancellation workflow. This task requires looking at subscription state, cancellation logs, billing policy, VIP context, urgent billing routing, a refund, and a goodwill credit.

The difficulty progression is meaningful:

- the easy task requires basic lookup and a single financial action
- the medium task requires multi-tag reasoning and more precise routing
- the hard task requires cross-artifact reasoning, VIP handling, and higher-cost policy judgment

## Grading

Each task has a deterministic grader that returns a score in `[0.0, 1.0]`.

The grader evaluates:

- whether the correct evidence was retrieved
- whether routing and priority are correct
- whether required tags were added
- whether the reply includes key required information
- whether the final resolution code and monetary values are correct

The scoring logic is deterministic, reproducible, and does not rely on randomness.

## Reward Design

The reward function is shaped across the full trajectory rather than being purely terminal.

Positive reward is assigned for:

- discovering required artifacts
- moving toward the correct route and priority
- improving tag coverage
- improving reply quality
- submitting a higher-quality final resolution

Penalties are applied for:

- repeated actions
- invalid actions
- hallucinated or unsupported lookups
- out-of-order API usage
- exceeding the maximum step budget
- acting after an episode is already complete

Per-step scalar reward is clipped into `[0.0, 1.0]`, while the richer penalty and progress decomposition is preserved in `reward_details`.

## Safety and Robustness

The environment is designed to fail safely:

- invalid task ids are rejected cleanly
- invalid actions return controlled errors
- `step()` cannot be used before `reset()`
- alias task ids such as `01`, `02`, `03`, `easy`, `medium`, and `hard` are supported
- episode boundaries are enforced through `done` and `max_steps`

## Project Structure

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
server/
  app.py
  Dockerfile
scripts/
  run_baseline.py
  validate-submission.sh
tests/
  test_environment.py
  test_server_api.py
inference.py
openenv.yaml
Dockerfile
uv.lock
```

## Local Development

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Validate the environment:

```bash
openenv validate . --json
```

Run the API locally:

```bash
uvicorn support_queue_env.server.app:app --host 0.0.0.0 --port 7860
```

Useful endpoints:

- `/docs`
- `/health`
- `/metadata`
- `/reset`
- `/step`
- `/state`

## Docker

Build and run locally:

```bash
docker build -t support-queue-openenv .
docker run --rm -p 7860:7860 support-queue-openenv
```

Smoke test:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/metadata
```

## Hugging Face Spaces Deployment

This repository is prepared for a Docker-based Hugging Face Space. The front matter at the top of this README sets:

- `sdk: docker`
- `app_port: 7860`
- `openenv` tag metadata

To deploy:

1. Create a new Hugging Face Space with SDK set to `Docker`
2. Push this repository to the Space
3. Wait for the image build to complete
4. Verify:
   - `https://<space>.hf.space/health`
   - `https://<space>.hf.space/docs`
   - `https://<space>.hf.space/metadata`

## Inference

The required inference entrypoint is `inference.py`.

It reads:

- `OPENAI_API_KEY`
- `MODEL_NAME`
- `API_BASE_URL`

Example:

```bash
export OPENAI_API_KEY=...
export MODEL_NAME=gpt-4.1-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

The script is deterministic by construction:

- fixed task order
- deterministic environment state
- deterministic graders
- `temperature=0`
- `seed=0`

Expected logging format:

```text
[START] Task=<task_id>
[STEP] reward=<float> done=<bool>
[END] total_reward=<float>
```

## Submission Validation

After deploying to Hugging Face Spaces, you can run the included validator:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space-name.hf.space .
```

This checks:

- the live Space responds on `/reset`
- Docker builds successfully
- `openenv validate` passes

## Reproducibility

Reproducibility is supported by:

- deterministic task fixtures
- deterministic grader logic
- checked-in `uv.lock`
- fixed task order in `inference.py`
- deterministic model invocation settings

## Baseline Results

The repository includes a reproducible inference runner, but baseline execution is not included in this README because it depends on external model credentials. Once valid API credentials are configured, `python inference.py` will produce deterministic per-task logs and aggregate outcomes suitable for benchmarking.
