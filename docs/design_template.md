# Design Template — Multi-Agent Research System

## Problem

Người dùng cần một research assistant có thể nhận câu hỏi phức tạp (ví dụ: "Summarise the state-of-the-art of GraphRAG"), tìm kiếm thông tin từ nhiều nguồn, phân tích chứng cứ, rồi trả về câu trả lời ~500 từ với citation rõ ràng cho đối tượng kỹ thuật.

Hệ thống phải đáp ứng:
- Tìm kiếm web / internal docs.
- Phân tích và so sánh viewpoints.
- Viết câu trả lời có citations.
- Không chạy vô hạn; có trace để debug.

## Why multi-agent?

Single-agent không đủ vì:

1. **Context window bị saturate**: nếu nhét search + analysis + writing vào 1 prompt, context limit bị vượt với query phức tạp.
2. **Separation of concerns**: researcher cần prompt khác với analyst và writer — trộn lẫn làm chất lượng giảm.
3. **Modularity / testability**: mỗi agent có thể test độc lập, swap implementation mà không ảnh hưởng agent khác.
4. **Parallel potential**: researcher và critic có thể chạy song song trong tương lai.
5. **Benchmark evidence**: kết quả benchmark trong `reports/benchmark_report.md` cho thấy multi-agent đạt quality score cao hơn single-agent trên query nghiên cứu phức tạp.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối workflow, quyết định agent nào chạy tiếp, enforce guardrails | `ResearchState` (toàn bộ) | `route_history` + `iteration` updated | Max-iterations exceeded → force `done` với fallback answer |
| Researcher | Tìm kiếm sources (Tavily / mock), tổng hợp research notes có citation | `request.query`, `request.max_sources` | `state.sources`, `state.research_notes` | Search API down → mock fallback; LLM error → retry 3 lần |
| Analyst | Trích xuất key claims, so sánh viewpoints, đánh giá evidence strength, chỉ ra gaps | `state.research_notes` | `state.analysis_notes` | Thiếu research_notes → `AgentExecutionError`; LLM error → retry |
| Writer | Viết final answer ~500 từ với inline citations và References section | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | Thiếu research_notes → `AgentExecutionError`; tạo skeleton answer nếu analysis_notes thiếu |

## Shared state

```python
class ResearchState(BaseModel):
    request: ResearchQuery        # query gốc, max_sources, audience
    iteration: int                # đếm số lần supervisor chạy
    route_history: list[str]      # ["researcher", "analyst", "writer", "done"]

    sources: list[SourceDocument] # documents từ search (title, url, snippet)
    research_notes: str | None    # tổng hợp từ researcher
    analysis_notes: str | None    # key claims + gaps từ analyst
    final_answer: str | None      # output cuối từ writer

    agent_results: list[AgentResult]  # mỗi agent push kết quả vào đây (token, cost)
    trace: list[dict]             # structured trace events cho debug
    errors: list[str]             # lỗi mỗi agent gặp phải
```

**Lý do từng field:**
- `iteration` + `route_history`: supervisor cần biết đã đi qua đâu để quyết định next step và enforce max_iterations.
- `sources`: writer cần để tạo references section.
- `agent_results`: benchmark cần để tính total token cost.
- `trace`: debug / LangSmith export.
- `errors`: tránh silent failure.

## Routing policy

```
START
  │
  ▼
Supervisor
  ├─ research_notes missing? ──────────────→ Researcher ─┐
  ├─ analysis_notes missing (research done)? → Analyst  ─┤
  ├─ final_answer missing (analysis done)? ──→ Writer   ─┘
  ├─ final_answer present? ──────────────────→ DONE
  └─ iteration ≥ max_iterations? ────────────→ DONE (fallback answer)
```

Supervisor chạy lại sau mỗi worker; mỗi worker push kết quả vào state rồi trả state về graph.

## Guardrails

- **Max iterations**: `MAX_ITERATIONS=6` (env). Supervisor force `done` nếu vượt.
- **Timeout**: `TIMEOUT_SECONDS=60` (env). OpenAI client timeout=60s per call.
- **Retry**: tenacity `@retry(stop_after_attempt(3), wait_exponential(min=2, max=10))` trong `LLMClient.complete()`.
- **Fallback**: SearchClient fallback về mock nếu Tavily down. Supervisor tạo partial answer nếu max_iterations đạt.
- **Validation**: Pydantic schema validates mọi input/output. `AgentExecutionError` nếu preconditions không đủ.

## Benchmark plan

**Query test:**
1. `"Research GraphRAG state-of-the-art and write a 500-word summary"`
2. `"What are the trade-offs of multi-agent vs single-agent LLM systems?"`

**Metrics:**

| Metric | Cách đo | Expected |
|---|---|---|
| Latency | wall-clock time (perf_counter) | single < multi, nhưng multi quality cao hơn |
| Cost | sum(token_cost) per agent_result | multi tốn hơn vì 3 LLM calls |
| Quality | rubric 0-10 tự động + peer review | multi ≥ 8, single ~5 |
| Citation coverage | `[Source N]` count in final_answer / num_sources | multi ≥ 80% |
| Failure rate | num errors / 1 run | 0 nếu API keys hợp lệ |

**Expected outcome:** Multi-agent đạt quality cao hơn ở cost cao hơn ~2-3x. Single-agent nhanh hơn nhưng thiếu citation và phân tích sâu.
