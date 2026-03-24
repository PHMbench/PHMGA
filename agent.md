# OpenRouter Live Run Notes

## Proxy Handling

OpenRouter live runs must ignore system proxy variables unless a proxy is intentionally part of the experiment.

Standard command form:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python main.py case_exp_ottawa
```

Why:

- `HTTP_PROXY` / `HTTPS_PROXY` were inherited by the runtime in live verification.
- The first failed Ottawa live run was routed through `127.0.0.1:8888` instead of going directly to OpenRouter.
- This polluted transport behavior before DAG construction finished.

The runtime now defaults to `trust_env=False` for OpenRouter requests, but the shell command above remains the safest live-run entrypoint.

## Fast Checks

Confirm `.env` is loaded and `OPENROUTER_API_KEY` is visible:

```bash
python - <<'PY'
from dotenv import load_dotenv
import os
load_dotenv()
print(bool(os.getenv("OPENROUTER_API_KEY")))
PY
```

Inspect whether a running process still inherited proxy variables:

```bash
tr '\0' '\n' < /proc/<PID>/environ | egrep '(^|_)(HTTP|HTTPS|ALL)_PROXY=|NO_PROXY='
```

Check whether the process is directly connected to OpenRouter over `:443`:

```bash
lsof -p <PID> | egrep 'TCP|IPv4|IPv6'
```

## Failure Triage

If `python main.py <case>` starts but produces no state or report artifacts:

- Check proxy inheritance first.
- If the process is still using a system proxy, treat transport as contaminated before debugging planner behavior.
- If the process is directly connected to OpenRouter and still waits a long time, prefer blaming the transport/read phase and repair retry, not the dataset layer.

If `dag.json` is missing entirely:

- The problem is still upstream of DAG materialization.
- The likely causes are planner transport latency, malformed provider output, or the follow-up repair request.

## Why HTTP Responses Can Be Slow

The slow path is not just "network is slow". It is the combination of:

1. Proxy pollution: the request may be routed through an unintended local proxy.
2. Coarse synchronous response handling: the client waits for the full response body before parsing JSON.
3. Repair amplification: if the first planner response is not valid JSON, the runtime immediately sends a second full repair request.

That means a single planner step can become two long blocking HTTP reads even before any DAG artifact is written.
