# AgentPhone Integration

AgentPhone is the live voice/SMS ingress path for COSMIC Browser Memory.

It lets a user call the agent, speak a browser task, and route that task into the COSMIC workflow stack. For the hackathon demo, this folder is intentionally lightweight: it focuses on provisioning, webhook handling, transcript streaming, SMS, and outbound call testing.

## What This Enables

```text
User calls agent
  -> AgentPhone webhook receives transcript
  -> COSMIC receives browser task
  -> cosmic-browser-use browses the web
  -> result can be spoken, texted, or attached through a follow-up channel
```

Example spoken request:

```text
Get me the latest 1040NR tax form.
```

COSMIC can then use its browser agent to find the form, complete the web task, and return the result.

## Files

| File | Purpose |
|---|---|
| `provision.py` | Create or attach AgentPhone resources and print the webhook signing secret. |
| `webhook_server.py` | FastAPI webhook server for SMS and voice webhook mode. |
| `make_call.py` | Outbound call test client with optional live transcript streaming. |
| `send_sms.py` | Send an SMS through the configured AgentPhone number. |
| `call_sse_transcript.py` | Subscribe to call transcript Server-Sent Events. |
| `security.py` | Verify AgentPhone webhook signatures. |
| `config.py` | Load required AgentPhone environment variables. |

## Setup

```powershell
cd AgentPhone
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env`:

```bash
AGENTPHONE_API_KEY=
AGENTPHONE_WEBHOOK_SECRET=
WEBHOOK_PUBLIC_BASE=
AGENTPHONE_PHONE_NUMBER=
AGENTPHONE_AGENT_ID=
AGENTPHONE_NUMBER_ID=
```

## Run the Webhook Server

Expose the server publicly with a tunnel such as ngrok, then run:

```powershell
uvicorn webhook_server:app --host 0.0.0.0 --port 8765
```

Provision or update AgentPhone resources:

```powershell
python provision.py
```

## Test a Call

```powershell
python make_call.py --list-from-numbers
python make_call.py --dry-run +15551234567
python make_call.py +15551234567
```

## Security Notes

- Never commit `.env`.
- Rotate any webhook secret that was pasted into a public place.
- `webhook_server.py` verifies AgentPhone signatures with `security.py`.

