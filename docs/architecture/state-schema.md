# Shared State Schema

The shared state schema is the Phase 1 contract between all future agents. After Phase 1, changes to this contract require a documented schema version update and review of every agent that reads or writes the changed field.

## PipelineStatus

| Value | Meaning |
|---|---|
| `idle` | Pipeline initialized, no request currently running. |
| `running` | A fact-check request is active. |
| `done` | Pipeline completed successfully and `final_report` is available. |
| `error` | An unrecoverable error occurred and `error` contains the reason. |

## ClaimResult

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | Exact extracted claim text. |
| `verdict` | `SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE` | Verdict based on retrieved evidence. |
| `confidence` | `float` | Normalized confidence score from `0.0` to `1.0`. |
| `evidence` | `list[str]` | Evidence snippets given to the verifier. |
| `sources` | `list[str]` | Source URLs corresponding to evidence snippets. |
| `reasoning` | `str` | Explanation connecting evidence to verdict. |
| `search_queries` | `list[str]` | Exact web search queries issued for the claim. |

## FactCheckState

| Field | Type | Owner | Description |
|---|---|---|---|
| `raw_input` | `str` | User Input | Original user-submitted text. |
| `extracted_claims` | `list[str]` | Extractor | Atomic, self-contained factual claims. |
| `claim_results` | `list[ClaimResult]` | Verifier | One result per extracted claim, populated incrementally. |
| `final_report` | `str | None` | Reporter | Final markdown report. |
| `messages` | `Annotated[list[BaseMessage], add_messages]` | Dialogue + User | Conversation history with append semantics. |
| `current_agent` | `str` | Orchestrator | Identifier of the active agent. |
| `session_id` | `str` | Orchestrator | UUID used for later persistence. |
| `error` | `str | None` | All agents | Describes unrecoverable failures. |
| `status` | `PipelineStatus` | Orchestrator | Current pipeline state. |
