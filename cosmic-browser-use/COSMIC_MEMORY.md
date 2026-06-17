# COSMIC Browser Memory

## Indexing the Web for Agents

COSMIC Browser Memory adds cross-run traversal memory around the existing vision browser agent.

Search engines indexed web pages for humans. COSMIC indexes how agents move through websites: page states, screenshots, visual anchors, actions, coordinates, failures, fixes, and successful workflows.

It runs on top of `cosmic-browser-use`, a vision-dominant browser automation agent that can see and operate real websites. The vision grounding model is **MiMo-7B-RL / MiMo-VL-7B-RL**.

On the first run, the agent completes a task normally while COSMIC records the traversal path. After the task succeeds, COSMIC compresses that run into a reusable workflow, stores the executable route locally, and writes semantic workflow memory to Supermemory.

On future runs, the agent retrieves relevant prior workflows, loads the executable route, adapts task-specific values, replays safe known actions, and hands control back to live vision whenever the website has changed or confidence drops.

The result is a browser agent that does not rediscover the same websites from scratch every time.

Every successful run becomes reusable intelligence for future tasks.

The normal agent still works with:

```powershell
python main.py --provider fireworks_kimi --goal "Get the YouTube video description for <video or query>"
```

## What Gets Stored

MiMo-7B-RL is prompted to return the center of a visual target as pixel coordinates:

```text
Image size: <w>x<h> pixels
Find the element: <instruction>
Output the center coordinates as [x, y] in pixels.
```

`browser_controller.parse_coordinates(...)` accepts explicit MiMo coordinate outputs like `[x, y]`, `(x, y)`, `x=... y=...`, JSON `{"x": ..., "y": ...}`, or JSON `position`/`coordinates` fields. If the model returns 0-1 coordinates, the parser converts them to pixels. It deliberately refuses arbitrary numbers from prose or truncated `<think>` blocks, so malformed reasoning triggers strict retry instead of clicking a fake coordinate.

COSMIC then stores a visual index for replay:

```json
{
  "target_description": "YouTube search box",
  "pixel_coordinates": {"x": 731, "y": 92},
  "normalized_coordinates": {"x": 0.571094, "y": 0.127778},
  "viewport": {"width": 1280, "height": 720}
}
```

Replay uses normalized coordinates against the current viewport, so the next run can click/type without another MiMo call.

## LLM Run Indexing

The run log is the raw evidence, not the final memory. After a successful
`learn`/`auto` run, COSMIC now sends a compact trace to the run indexer LLM. The
indexer produces a gold-path workflow:

- selects only reusable source steps;
- discards wrong clicks, retries, waits, and dead-end pages;
- preserves each selected step's source-step evidence;
- keeps MiMo visual indexes exactly as recorded;
- separates stored evidence from replay policy: unsafe/dynamic coordinates stay
  in `source_visual_index`, while only replay-safe coordinates are exposed as
  `visual_index`;
- parameterizes dynamic fields such as search queries;
- marks checkpoints where the normal agent should re-plan;
- records failure patches and rejected detours.

The indexer is forbidden to invent coordinates. Visual replay coordinates always
come from the original MiMo-grounded source step:

```text
run log -> LLM distiller -> workflow JSON -> local store + Supermemory summary
```

If the LLM indexer is unavailable, COSMIC falls back to a deterministic compiler
that preserves successful replayable actions and observed final URLs.

The workflow also keeps compact `trace_evidence` for every action in the run,
including discarded detours and failed attempts. This lets later indexers learn
from the whole successful run without forcing replay to execute unsafe steps.

Useful env vars:

```bash
COSMIC_INDEXER_ENABLED=true
COSMIC_INDEXER_PROVIDER=fireworks_kimi
COSMIC_INDEXER_MODEL=accounts/fireworks/models/kimi-k2p6
COSMIC_INDEXER_MAX_TOKENS=4096
```

## Modes

`off`: original browser agent behavior.

`learn`: run normally, then compile the run into `data/cosmic_memory/workflows/*.json` and write a semantic summary to Supermemory when configured.

`recall`: retrieve an existing workflow, build one upfront replay plan, execute that plan without per-step LLM calls until checkpoint, then continue with the normal agent. If the indexed workflow has an observed successful final URL and the current goal strongly matches that target, recall uses the deterministic observed-final-URL fast path before spending a slow planner call.

`auto`: recall first, then compile the finished run back into memory.

## Real-Site Workflow Rehearsal

Index a workflow:

```powershell
python main.py --provider fireworks_kimi --memory-mode learn --goal "Get the YouTube video description for the official OpenAI GPT-4o launch video" --demo-overlay
```

Index an already-completed run:

```powershell
python scripts/index_run.py --run-dir runs/20260517_155250
```

For a local-only compile without Supermemory:

```powershell
python scripts/index_run.py --run-dir runs/20260517_155250 --disable-supermemory
```

Replay/adapt it:

```powershell
python main.py --provider fireworks_kimi --memory-mode recall --goal "Get the YouTube video description for the official Claude Code launch video" --demo-overlay
```

For an exact target repeat, replay can skip search/result selection entirely:

```text
recall -> observed final URL -> indexed scroll/expand actions -> checkpoint finalizer
```

For a changed target, replay falls back to the safe prefix and checkpoints at the dynamic result-selection step.

Use `auto` during rehearsals when you want recall plus writeback:

```powershell
python main.py --provider fireworks_kimi --memory-mode auto --goal "Get the YouTube video description for <new video>" --demo-overlay
```

## Supermemory

Set `SUPERMEMORY_API_KEY` in `.env`. The Python SDK uses:

```python
from supermemory import Supermemory

client = Supermemory(api_key=os.environ["SUPERMEMORY_API_KEY"])
client.add(content="...", container_tag="cosmic-hackathon-demo:demo_user", task_type="memory")
client.search.memories(q="...", container_tag="cosmic-hackathon-demo:demo_user", search_mode="hybrid")
```

COSMIC keeps exact executable route memory in local JSON and uses Supermemory for semantic recall.

We explicitly use `task_type="memory"` for workflow summaries and retrieve with `search.memories(..., search_mode="hybrid")`. Supermemory documents `memory` as the full memory/context layer with SuperRAG built in, while `superrag` is the document/RAG-only mode. Browser traversal memory needs temporal context, user preferences, failure patches, and relationship-aware recall, so `memory` plus hybrid memory search is the right fit.

## AgentPhone and AgentMail Ingress

COSMIC can be invoked from more than the CLI.

`AgentPhone/` provides a live call/SMS integration path. A user can call the agent, speak a browser task, and route that task into COSMIC.

AgentMail follows the same task-ingress pattern for email. A user can email a task such as:

```text
Get me the latest 1040NR tax form.
```

The agent browses, completes the task, and replies with the result or attachment through the email layer.

## Debug Logs

Each run writes structured JSONL logs under:

```text
runs/<timestamp>/cosmic_debug/
  events.jsonl
  steps.jsonl
  memory.jsonl
  replay.jsonl
```

Use these to debug:

- `steps.jsonl`: capture, LLM decision, action execution, verification, persisted visual index.
- `memory.jsonl`: runtime initialization, recall probes, retrieval scores, indexed workflow writeback.
- `replay.jsonl`: planner result, indexed action before/after states, replay failures/checkpoints.
- `events.jsonl`: run start/final stats and high-level lifecycle events.
- `indexer_plan.json`: compact trace summary plus the LLM/fallback plan used to build the workflow.

## Supermemory Smoke Test

After setting `SUPERMEMORY_API_KEY`, run:

```bash
python scripts/smoke_supermemory.py
```

The script writes an isolated test workflow using `task_type="memory"`, retries search until the marker is retrieved, and saves a report under:

```text
data/smoke_supermemory/
```
