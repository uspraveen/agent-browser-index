<div align="center">
  <img src="assets/cosmic-ball-logo-v1.1.png" alt="Cosmic Browser Agent" width="120" />
  <h1>Cosmic Browser Agent</h1>
  <p><strong>Vision-dominant browser automation for real websites — with optional traversal memory.</strong></p>
</div>

Cosmic Browser Agent is an autonomous web task runner that sees and operates real sites through Playwright, MiMo-VL visual grounding, and an LLM orchestrator. **COSMIC Browser Memory** is the optional layer on top: it records how the agent moves through websites and replays successful routes on future runs.

Search engines indexed pages for humans. Cosmic indexes **how agents traverse sites**.

---

## Why This Exists

Most browser agents are stateless. Each run rediscovers the same UI: where the search bar is, which button opens the result, what failed last time, and where the successful path was.

Cosmic Browser Agent solves the execution problem with vision-grounded actions. COSMIC memory solves the repetition problem:

1. **Learn** — complete a task while COSMIC records the traversal path.
2. **Index** — distill the successful run into a replayable workflow stored locally.
3. **Recall** — retrieve a prior workflow, replay safe known actions, and use live vision only where the site changed or confidence drops.

```plaintext
First run:   explore → complete task → index route
Future run:  recall route → replay known actions → live vision only where needed
```

---

## What COSMIC Memory Stores

COSMIC memory stores executable traversal intelligence, not just text summaries.

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

Cosmic Browser Agent runs from **`cosmic-browser-use`**, with COSMIC memory as an optional subsystem.

| Component | Role |
|---|---|
| **Playwright** | Browser control and action execution |
| **MiMo-VL-7B-RL** | Screenshot → pixel coordinate visual grounding |
| **Provider LLM** | Planning, adaptation, and run indexing (Fireworks Kimi by default) |
| **Local workflow store** | Deterministic replay from JSON workflows |
| **Supermemory** *(optional)* | Semantic recall across workflows when `SUPERMEMORY_API_KEY` is set |

```plaintext
User task
    │
    ▼
Workflow retriever ──► Local COSMIC workflow store
    │                           │
    │                           ▼
    │                    Replay planner
    │                           │
    │                           ▼
    └──────────────► cosmic-browser-use
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    MiMo-VL grounding   Playwright/Chrome   Action recorder
                                               │
                                               ▼
                                        Workflow indexer
                                               │
                                               ▼
                                    Local COSMIC workflow store

Optional: Supermemory semantic recall (when SUPERMEMORY_API_KEY is set)
```

**Mental model:**

```plaintext
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

```env
FIREWORKS_API_KEY=your_key_here
MIMO_API_URL=https://your-mimo-server.modal.run
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

## COSMIC Memory Modes

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

For sites that require existing logins (LinkedIn, Google accounts, etc.), the agent can launch your real Chrome binary against a **dedicated agent copy** of a profile — your live Chrome windows are never touched.

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

```plaintext
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

COSMIC memory works fully with **local workflow storage only**. Supermemory is an optional semantic recall layer for finding related workflows across large memory sets.

Enable by setting `SUPERMEMORY_API_KEY` in `.env`. Disable explicitly with `--disable-supermemory`.

```bash
# Verify Supermemory connectivity
python scripts/smoke_supermemory.py
```

> Local COSMIC storage holds the route. Supermemory holds the context — when you choose to use it.

---

## Repository Layout

```plaintext
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
