<p align="center">
  <img src="docs/assets/cosmic-ball-logo-v1.1.png" alt="COSMIC Browser Memory" width="160" />
</p>

# COSMIC Browser Memory

**Traversal memory for browser agents — learn a route once, replay it on future runs.**

COSMIC Browser Memory records how an agent moves through real websites: page states, screenshots, visual anchors, actions, coordinates, failures, fixes, and successful workflows. Every successful run can become reusable intelligence instead of starting from scratch.

Search engines indexed pages for humans. COSMIC indexes **how agents traverse sites**.

---

## Why This Exists

Most browser agents are stateless. Each run rediscovers the same UI: where the search bar is, which button opens the result, what failed last time, and where the successful path was.

COSMIC changes that:

1. **Learn** — complete a task while COSMIC records the traversal path.
2. **Index** — distill the successful run into a replayable workflow stored locally.
3. **Recall** — retrieve a prior workflow, replay safe known actions, and use live vision only where the site changed or confidence drops.

```text
First run:   explore → complete task → index route
Future run:  recall route → replay known actions → live vision only where needed
```

---

## What COSMIC Stores

COSMIC stores executable traversal intelligence, not just text summaries.

| Memory type | What gets stored |
|---|---|
| Page states | URL, title, viewport, scroll position, screenshots |
| Visual indexes | MiMo-grounded pixel coordinates + normalized viewport coordinates |
| Actions | Navigate, click, type, scroll, expand, extract, save |
| Workflows | Ordered successful paths with checkpoints and replay policies |
| Failures | Bad clicks, wrong targets, loops, dead ends |
| Fixes | Failure patches and replay guardrails |

```json
{
  "workflow_id": "youtube_video_description_success",
  "domain": "youtube.com",
  "intent": "Find a YouTube video and extract its description",
  "steps": [
    {
      "action_type": "VisualClick",
      "target_description": "YouTube search bar",
      "normalized_coordinates": {"x": 0.465625, "y": 0.038889},
      "replay_safe": true
    }
  ],
  "failure_patches": [
    {
      "failure": "Clicked wrong search result",
      "fix": "Verify title/channel before continuing to description extraction"
    }
  ]
}
```

---

## Architecture

COSMIC runs on top of **`cosmic-browser-use`**, a vision-dominant browser automation agent.

| Component | Role |
|---|---|
| **Playwright** | Browser control and action execution |
| **MiMo-VL-7B-RL** | Screenshot → pixel coordinate visual grounding |
| **Provider LLM** | Planning, adaptation, and run indexing (Fireworks Kimi by default) |
| **Local workflow store** | Deterministic replay from JSON workflows |
| **Supermemory** *(optional)* | Semantic recall across workflows when `SUPERMEMORY_API_KEY` is set |

```mermaid
flowchart TD
    U[User task] --> R[Workflow retriever]
    R --> W[Local COSMIC workflow store]
    W --> P[Replay planner]
    P --> B[cosmic-browser-use]
    B --> V[MiMo-VL visual grounding]
    B --> PW[Playwright / Chrome via CDP]
    B --> A[Action recorder]
    A --> C[Workflow indexer]
    C --> W
    C -.->|optional| SM[Supermemory semantic recall]
    R -.->|optional| SM
```

**Mental model:**

```text
Local COSMIC store  = executable traversal map (source of truth)
MiMo-VL             = visual grounding model
Playwright / Chrome = browser hands
Supermemory         = optional semantic recall layer (not required)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Google Chrome installed (required for `--chrome-profile`; optional otherwise)
- API keys for your chosen LLM provider and a reachable MiMo grounding server

### Install

```bash
cd cosmic-browser-use
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Edit `.env` with at least:

```bash
FIREWORKS_API_KEY=your_key_here
MIMO_API_URL=https://your-mimo-server.modal.run   # or self-hosted vLLM endpoint
```

### Run a task (no memory)

```bash
python main.py \
  --provider fireworks_kimi \
  --interaction-mode vision \
  --goal "Get the YouTube video description for the official OpenAI GPT-4o launch video"
```

### Learn a workflow

```bash
python main.py \
  --provider fireworks_kimi \
  --interaction-mode vision \
  --memory-mode learn \
  --goal "Get the YouTube video description for the official OpenAI GPT-4o launch video"
```

### Recall and adapt a workflow

```bash
python main.py \
  --provider fireworks_kimi \
  --interaction-mode vision \
  --memory-mode recall \
  --goal "Get the YouTube video description for the official Claude Code launch video"
```

### Compare against a memory-less baseline

```bash
python main.py \
  --provider fireworks_kimi \
  --interaction-mode vision \
  --memory-mode off \
  --goal "Get the YouTube video description for the official Claude Code launch video"
```

---

## Memory Modes

| Mode | Behavior |
|---|---|
| `off` | Standard vision browser agent. No cross-run memory. |
| `learn` | Run normally, then index the successful traversal into local COSMIC storage. |
| `recall` | Retrieve a prior workflow, replay safe actions, resume live planning at checkpoints. |
| `auto` | Recall first, then write the completed run back into memory. |

Set via `--memory-mode` or `COSMIC_MEMORY_MODE` in `.env`.

Local workflows are stored under `./data/cosmic_memory/` by default (`--memory-dir` / `COSMIC_MEMORY_DIR`).

---

## Chrome Profile Support

For sites that require existing logins (LinkedIn, Google accounts, etc.), COSMIC can launch your real Chrome binary against a **dedicated agent copy** of a profile — your live Chrome windows are never touched.

```bash
# List available profiles
python main.py --list-chrome-profiles

# Run with a seeded profile
python main.py \
  --provider fireworks_kimi \
  --chrome-profile "Default" \
  --goal "Search LinkedIn for software engineer roles in San Francisco"
```

| Flag | Purpose |
|---|---|
| `--chrome-profile` | Profile directory name (`Default`, `Profile 1`, …) or absolute path |
| `--refresh-chrome-profile` | Re-copy logins/cookies from your real profile into the agent dir |
| `--restore-tabs` | Best-effort reopen of URLs that were open in your live profile |

Agent profile data is stored under `%LOCALAPPDATA%\CosmicBrowserUse\chrome\` on Windows (or the platform equivalent). Override with `COSMIC_AGENT_DATA_DIR`.

---

## MiMo Vision Grounding

Visual tools (`VisualClick`, `VisualType`, `VisualHover`) send a screenshot plus target description to **MiMo-VL-7B-RL** and expect a center pixel coordinate back.

```text
Screenshot + "YouTube search bar at top center" → [596, 28]
```

The default endpoint is a Modal-hosted vLLM server. Deploy your own with the root-level `deploy_modal.py`:

```bash
modal deploy deploy_modal.py
```

Point `MIMO_API_URL` at your deployment base URL (no `/v1/chat/completions` suffix needed — the controller normalizes it). `main.py` runs a health pre-check before each task.

---

## Workflow Recording (Human-Driven)

Record a workflow by browsing manually instead of letting the agent explore:

```bash
python scripts/record_workflow.py \
  --workflow-name "LinkedIn job search" \
  --goal "Search LinkedIn for software engineer roles" \
  --chrome-profile "Profile 9"
```

The recorder captures DOM clicks, typing, scrolls, and navigation, then compiles the trace through the same indexing pipeline used by `learn` runs.

---

## Post-Run Indexing

Index an already-completed run directory:

```bash
python scripts/index_run.py --run-dir runs/20260517_155250
```

Local-only (skip Supermemory upload):

```bash
python scripts/index_run.py --run-dir runs/20260517_155250 --disable-supermemory
```

---

## Supermemory (Optional)

COSMIC works fully with **local workflow storage only**. Supermemory is an optional semantic recall layer for finding related workflows across large memory sets.

Enable by setting `SUPERMEMORY_API_KEY` in `.env`. Disable explicitly with `--disable-supermemory`.

```bash
# Verify Supermemory connectivity
python scripts/smoke_supermemory.py
```

> Local COSMIC storage holds the route. Supermemory holds the context — when you choose to use it.

---

## Repository Layout

```text
agent-browser-index/
  deploy_modal.py                   # Modal deployment for MiMo-VL vLLM server
  cosmic-browser-use/
    main.py                         # CLI entry point and run loop
    browser_controller.py           # Playwright, MiMo grounding, Chrome CDP
    orchestrator.py                 # LLM planning and replay adaptation
    memory_manager.py               # In-run memory and summaries
    cosmic_memory/
      replay.py                     # Indexed replay execution
      indexer.py                    # LLM run distiller → gold-path workflow
      trace_compiler.py             # Trace → workflow compiler (fallback)
      supermemory_client.py         # Optional Supermemory bridge
      recorder.py                   # Human-driven workflow recorder
    scripts/
      record_workflow.py            # Record a human browsing session
      index_run.py                  # Index an existing run directory
      smoke_supermemory.py          # Supermemory connectivity test
```

---

## Configuration Reference

See `cosmic-browser-use/.env.example` for the full list. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | — | LLM provider key (default provider) |
| `MIMO_API_URL` | Modal endpoint | MiMo grounding server base URL |
| `COSMIC_MEMORY_MODE` | `off` | Default memory mode |
| `COSMIC_MEMORY_DIR` | `./data/cosmic_memory` | Local workflow store |
| `COSMIC_INTERACTION_MODE` | `hybrid` | `hybrid` (DOM + vision) or `vision` (vision only) |
| `SUPERMEMORY_API_KEY` | — | Optional semantic recall |
| `COSMIC_INDEXER_ENABLED` | `true` | LLM run distiller for gold-path workflows |

---

## Further Reading

- [`cosmic-browser-use/README.md`](cosmic-browser-use/README.md) — browser agent internals, full tool surface, runtime loop, SDK usage
- [`cosmic-browser-use/COSMIC_MEMORY.md`](cosmic-browser-use/COSMIC_MEMORY.md) — memory modes, indexing pipeline, debug logs, replay behavior

---

## Production Notes

- **Secrets:** keep all API keys in `.env` or a secret manager. Never commit real keys.
- **MiMo is required:** this is a vision-dominant system; the agent cannot function without a reachable MiMo server.
- **Chrome profile mode** seeds a copy of your profile — close Chrome before `--refresh-chrome-profile` for a complete cookie sync.
- **Replay is best-effort:** sites change. COSMIC checkpoints and falls back to live vision when confidence drops.
- **No automated test suite yet** — validate changes with smoke runs against your target sites.
