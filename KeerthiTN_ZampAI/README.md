# Personalized Outreach Engine (PS-3)

Takes a prospect name + company, researches public signal, and produces a
grounded outreach draft for human review — or explicitly flags when it
shouldn't guess.

## Stack (free / open source only)
- **Search:** `duckduckgo-search`, Wikipedia REST API, HN Algolia Search API, GDELT Doc API, Reddit `search.json` — all free, no API keys
- **LLM:** [Ollama](https://ollama.com) running a local open model — free, open source, no key
- **Backend:** Flask + SQLite
- **Frontend:** plain HTML/JS (no build step)

## Setup

```bash
cd ps3-reachout
pip install -r requirements.txt --break-system-packages

# Optional but recommended — real LLM instead of the rule-based fallback:
# 1. Install Ollama: https://ollama.com/download
# 2. ollama pull llama3.2
# 3. Ollama usually runs as a background service after install
# 4. create a virtual environment if running locally

python app.py
```

Open **http://localhost:5000**. Fill in a prospect name + company, hit **Run
process**, and watch the live stage log. Every run is saved and shown in the
dashboard table below.

> Note: if Ollama isn't installed/running, the app automatically falls back
> to a simple rule-based extractor/writer so the pipeline still runs
> end-to-end — you'll see `[llm] ... fell back to mock` in the console.
> Swap in Ollama any time without changing the UI or flow.

## How it works

1. **Research** — `research.py` queries DuckDuckGo news + text search for the
   prospect/company combo.
2. **Signal extraction** — `llm.py` asks the model to rank candidate hooks
   with a confidence score and sentiment tag.
3. **Business rules** (in `app.py`, deliberately kept out of the LLM's hands):
   - discard negative-sentiment hooks (layoffs, lawsuits, etc.)
   - discard low-confidence hooks
   - flag hooks older than 180 days as stale
   - flag weak prospect identification (research barely mentions the person + company together)
   - if nothing survives → `insufficient_signal`, no draft fabricated
4. **Draft generation** — writes a short, grounded email from the surviving hook.
5. **Output** — saved to SQLite, shown in the dashboard, always labeled for
   human review.

## Edge cases built in (see `app.py::run_pipeline`)
| Edge case | What happens |
|---|---|
| No signal found at all | `insufficient_signal` status, no fabricated draft |
| Best signal is negative news **about the company** | Discarded, falls through to next candidate |
| Negative signal is **about the prospect's employment** (laid off, departed) | Hard stop — `prospect_may_have_departed`. Different in kind from company PR: the whole email premise is wrong, so it isn't fixed by picking a different hook |
| Prospect name can't be verified by any source, even though the company is real (e.g. a fabricated name at a real company) | Hard stop — `unverified_person`. Company-only evidence used to be enough to clear the identification threshold with just a soft flag; now checked as its own gate |
| Signal is old (>180 days) | Draft still produced, flagged `stale_signal` |

## Suggested demo script (for the 5-min video / live interview)
1. Run a **happy path**: a real, findable exec at a real company with recent
   news — show the live stage log and the resulting draft.
2. Run the **negative-news case**: pick a company you know had layoffs or
   controversy recently — show the discard trace in the business_rules stage.
3. Run the **insufficient signal case**: a made-up name at a tiny/unknown
   company — show it refuses to fabricate and flags instead.
4. Glance at the dashboard to show run history persists.

## Extending this
- Swap `duckduckgo-search` for a richer source (company blog RSS, Crunchbase
  free tier, etc.) by editing `research.gather_signal`.
- Swap Ollama's `llama3.2` for a bigger local model, or point `llm.py` at any
  API by changing `_call_ollama`.
- Add auth/multi-user support to SQLite if this needs to be shared.
- Make it an Agentic flow with MCPs which automates searching the 200 targets and personalizes messages for each separately.
- Make the UI more intuitive with clickable, hover sections and less technical Jargon.
- Compartmentalize behavior, responses for further security.
