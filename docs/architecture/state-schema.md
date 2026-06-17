# Shared State Schema

The shared state schema is the contract between all implemented agents. Changes to this schema require review of every agent, API model, and frontend type that reads or writes the affected field.

## PipelineStatus

| Value | Meaning |
|---|---|
| `idle` | Pipeline initialized, no request currently running. |
| `running` | A fact-check request is active. |
| `done` | Pipeline completed successfully and `final_report` is available. |
| `error` | An unrecoverable error occurred and `error` contains the reason. |

## ProcessingStatus

Optional field on `ClaimResult.processing_status`. Omitted when processing succeeds (`ok`).

| Value | Meaning |
|---|---|
| `ok` | Verifier completed normally (field omitted from serialized results). |
| `error` | Verifier failed for this claim; see `processing_error`. |
| `degraded` | Verifier returned a verdict but with reduced confidence or partial failure. |

## ClaimResult

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | Exact extracted claim text. |
| `verdict` | `SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE | CONFLICTING_EVIDENCE` | Verdict based on retrieved evidence. |
| `confidence` | `float` | Normalized confidence score from `0.0` to `1.0`. |
| `evidence` | `list[str]` | Evidence text excerpts presented to the verifier. Internally sourced from full-page HTTP fetches (top-ranked hits) or search-result snippets (fallback). |
| `sources` | `list[str]` | Source URLs corresponding to evidence excerpts. |
| `reasoning` | `str` | Explanation connecting evidence to verdict. |
| `search_queries` | `list[str]` | Exact web search queries issued for the claim. |
| `source_sentence` | `str \| null` | Original sentence the claim was extracted from. Optional (`NotRequired`). |
| `fidelity_status` | `str \| null` | Extractor fidelity outcome (`faithful` or `fallback`). Optional (`NotRequired`). |
| `processing_status` | `ok \| error \| degraded` | Verifier processing outcome. Optional (`NotRequired`); omitted when `ok`. |
| `processing_error` | `str \| null` | Error detail when `processing_status` is `error`. Optional (`NotRequired`). |

Evidence hit pre-filtering uses Okapi BM25 re-ranking over each search result set before the evaluator LLM is invoked. Source credibility uses static domain-tier heuristics (high/medium/low) applied during hit re-ranking and exposed to the evaluator; this is not a full reputation database.

## FactCheckState

| Field | Type | Owner | Description |
|---|---|---|---|
| `raw_input` | `str` | User Input | Original user-submitted text. |
| `extraction_mode` | `auto \| claim \| document` | API / pipeline | How the extractor interprets input. Optional (`NotRequired`). |
| `extracted_claims` | `list[ValidatedClaim]` | Extractor | Atomic, self-contained factual claims with original sentence context. |
| `claim_results` | `list[ClaimResult]` | Verifier | One result per extracted claim, populated in a single parallel verifier invocation (order matches `extracted_claims`). |
| `final_report` | `str | None` | Reporter | Final markdown report. |
| `messages` | `Annotated[list[BaseMessage], add_messages]` | Dialogue + User | Conversation history with append semantics. |
| `current_agent` | `str` | Orchestrator | Identifier of the active agent. |
| `session_id` | `str` | Orchestrator | UUID used for later persistence. |
| `error` | `str | None` | All agents | Describes unrecoverable failures. |
| `status` | `PipelineStatus` | Orchestrator | Current pipeline state. |
