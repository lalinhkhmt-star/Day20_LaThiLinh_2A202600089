# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| single-agent-baseline | 17.90 | 0.0002 | 5.0 |  |
| multi-agent-workflow | 30.58 | 0.0007 | 10.0 |  |

## Multi-Agent Trace

```json
[
  {
    "name": "supervisor.routed",
    "payload": {
      "route": "researcher",
      "iteration": 1
    }
  },
  {
    "name": "researcher.done",
    "payload": {
      "num_sources": 5,
      "notes_length": 1136
    }
  },
  {
    "name": "supervisor.routed",
    "payload": {
      "route": "analyst",
      "iteration": 2
    }
  },
  {
    "name": "analyst.done",
    "payload": {
      "analysis_length": 683
    }
  },
  {
    "name": "supervisor.routed",
    "payload": {
      "route": "writer",
      "iteration": 3
    }
  },
  {
    "name": "writer.done",
    "payload": {
      "answer_length": 1695
    }
  },
  {
    "name": "supervisor.routed",
    "payload": {
      "route": "done",
      "iteration": 4
    }
  }
]
```

## Failure Modes & Analysis

### Observed failure modes
- **Timeout / rate-limit**: LLM API rate-limit causes retry delays → mitigated by tenacity retry with exponential back-off.
- **Empty sources**: If Tavily is unavailable, mock results are used → quality degrades but does not crash.
- **Max-iteration fallback**: Supervisor forces `done` after `MAX_ITERATIONS` to prevent infinite loops.

### Fix applied
- Retry logic in `LLMClient` (3 attempts).
- Mock fallback in `SearchClient`.
- Hard cap in `SupervisorAgent` (`max_iterations`).
