"""
Stages: prospect summary, signal extraction, draft generation.
Uses a local, open-source model via Ollama (free, no API key). Falls back
to a deterministic mock if Ollama isn't running, so the pipeline always
completes end-to-end.

To use the real model:
  1. Install Ollama: https://ollama.com/download
  2. Run:  ollama pull llama3.2
  3. Ollama usually runs as a background service after install
"""
import json
import re
import difflib

MODEL_NAME = "llama3.2"

# Who "we" are as the sender. Per the case study brief, this tool is Zamp's
# own SDR outreach engine -- Zamp builds automated processes for messy
# operational work (invoice processing, vendor onboarding, GTM execution).
# Edit this if you want to point the tool at a different sender company --
# just don't also use that same company as your test PROSPECT below.
SENDER = {
    "company_name": "Zamp",
    "one_liner": "an AI solutions company that builds automated processes for real "
                 "operational work: invoice processing, vendor onboarding, GTM "
                 "execution: so that Ops and revenue teams get intelligence and "
                 "automation: without adding headcount.",
    "sender_name": "[Keerthi T N] (Zamp SDR)",
}

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

CATEGORY_LABELS = {
    "news": "Recent news",
    "hiring": "Hiring signal",
    "person_mention": "About this person",
    "company_overview": "Company overview",
    "manual_context": "Provided by you",
}


def _call_ollama(prompt: str, temperature: float = 0.2) -> str:
    """
    Low temperature (default 0.2) deliberately -- this call needs to return
    consistent, parseable JSON with sober confidence/sentiment judgments,
    not creative variation. num_predict caps runaway generation so a
    malformed prompt can't hang the pipeline waiting on token output.
    """
    resp = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": 700, "top_p": 0.9},
    )
    return resp["response"]


def _call_ollama_json(prompt: str, retries: int = 1):
    """
    Calls Ollama and parses JSON out of the response. On a parse failure,
    retries once with a stricter, more explicit instruction appended rather
    than immediately giving up -- local open models occasionally wrap JSON
    in prose or markdown fences on the first attempt.
    """
    last_err = None
    for attempt in range(retries + 1):
        p = prompt if attempt == 0 else (
            prompt + "\n\nIMPORTANT: reply with ONLY the raw JSON object. "
                      "No markdown fences, no commentary, no leading/trailing text."
        )
        try:
            raw = _call_ollama(p)
            return _extract_json(raw)
        except Exception as e:
            last_err = e
    raise last_err


def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


# ---------- Prospect summary (deterministic, no LLM needed -- fast & inspectable) ----------

def build_prospect_summary(company_name: str, bundle: dict, id_score: float, id_label: str):
    about = bundle.get("about_us")
    snapshot_verified = bool(about) and about.get("verified", True) and bool(about.get("description"))
    if snapshot_verified:
        company_snapshot = about["description"]
    elif about and not about.get("verified", True):
        company_snapshot = ("Found a page at this domain, but it doesn't appear to actually "
                             "describe this company -- possibly a different business with a "
                             "similar name. Not used as a hook.")
    else:
        company_snapshot = "No public overview found."

    top_signals = []
    for n in bundle.get("news", [])[:2]:
        top_signals.append({"category": "news", "text": n["title"], "url": n["url"], "date": n.get("date", "")})
    for h in bundle.get("hiring_signals", [])[:1]:
        top_signals.append({"category": "hiring", "text": h["title"], "url": h["url"], "date": ""})
    for m in bundle.get("person_mentions", [])[:1]:
        top_signals.append({"category": "person_mention", "text": m["title"], "url": m["url"], "date": ""})

    return {
        "company_name": company_name,
        "company_snapshot": company_snapshot[:280],
        "company_website": bundle.get("company_website"),
        "top_signals": top_signals,
        "identification_score": id_score,
        "identification_label": id_label,
    }


_CATEGORY_PRIOR = {  # rough prior on how useful a category tends to be as a hook
    "manual_context": 1.3,  # human-vetted, trumps anything scraped
    "news": 1.0, "hiring": 0.85, "person_mention": 0.8, "company_overview": 0.5,
}


def _prerank_candidates(prospect_name: str, company_name: str, candidates: list, keep: int = 8):
    """
    Cheap keyword-overlap + category-prior scoring pass BEFORE the LLM call.
    This is the "retrieval" half of the pipeline: instead of dumping every
    raw candidate (including weak, near-duplicate, or barely-relevant ones)
    into the prompt, only the top `keep` most promising candidates go in.
    Keeps the prompt focused, cuts noise the model has to reason around,
    and keeps latency/cost bounded regardless of how wide the research net
    gets (now up to 8 sources -- see research.py).
    """
    name_tokens = {p for p in prospect_name.lower().split() if len(p) > 2}
    company_tokens = {p for p in re.findall(r"[a-zA-Z]+", company_name.lower()) if len(p) > 2}

    scored = []
    for c in candidates:
        blob = f"{c['title']} {c['snippet']}".lower()
        blob_words = set(re.findall(r"[a-zA-Z]+", blob))
        overlap = len(blob_words & (name_tokens | company_tokens))
        specificity = min(len(c.get("snippet", "")) / 200.0, 1.0)  # longer snippet = more concrete, capped
        score = _CATEGORY_PRIOR.get(c["category"], 0.5) + 0.3 * overlap + 0.2 * specificity
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:keep]]


def _truncate_for_prompt(text: str, limit: int = 300) -> str:
    """Chunks each snippet down before it goes into the prompt -- long raw
    scrapes waste context and dilute the model's attention on the parts
    that actually matter for a one-sentence hook."""
    text = text or ""
    return text[:limit] + ("..." if len(text) > limit else "")


# ---------- Signal extraction ----------

def extract_signal(prospect_name: str, company_name: str, candidates: list):
    """
    Ranks candidate hooks (already tagged with category) by how good they'd
    be as a personalization hook. Returns ranked list, best first.
    """
    if not candidates:
        return []

    if not _OLLAMA_AVAILABLE:
        return _mock_extract_signal(candidates, company_name)

    # retrieval pass: narrow raw candidates down to the most promising ones
    # before they ever reach the model (see _prerank_candidates docstring)
    pool = _prerank_candidates(prospect_name, company_name, candidates, keep=8)

    snippets_text = "\n\n".join(
        f"[{i}] category={c['category']} date={c.get('date','')}\n"
        f"title: {c['title']}\nsnippet: {_truncate_for_prompt(c['snippet'])}"
        for i, c in enumerate(pool)
    )

    prompt = f"""You are helping a B2B sales rep find the best personalization
hook for a cold email to {prospect_name} at {company_name}.

Below are raw snippets from four categories: news, hiring, person_mention,
company_overview. Rank the top 3 as outreach hooks. For each, judge:
- confidence (0.0-1.0): how specific and genuinely useful this is
- sentiment: "positive", "neutral", or "negative" (negative = layoffs,
  lawsuits, scandal are inappropriate to reference in a sales pitch)
- hook_sentence: ONE natural sentence a human would say about this signal
  (not a copy of the title -- paraphrase it conversationally)
- reasoning: one sentence on why this would land well or not
- Avoid generic flattery, exclamation points, or "I hope this finds you well".
- Let the text be human and conversational, not a template. Don't quote verbatim -- paraphrase.

Snippets:
{snippets_text}

Respond with ONLY a JSON object:
{{"candidates": [
  {{"index": <int>, "hook_sentence": "<...>", "confidence": <float>,
    "sentiment": "<positive|neutral|negative>", "reasoning": "<...>"}}
]}}
"""
    try:
        parsed = _call_ollama_json(prompt)
        ranked = []
        for item in parsed.get("candidates", []):
            idx = item.get("index")
            if idx is None or idx >= len(pool):
                continue
            ranked.append({
                **pool[idx],
                "hook_sentence": item.get("hook_sentence", pool[idx]["title"]),
                "confidence": float(item.get("confidence", 0.3)),
                "sentiment": item.get("sentiment", "neutral"),
                "reasoning": item.get("reasoning", ""),
            })
        ranked.sort(key=lambda x: x["confidence"], reverse=True)
        return ranked or _mock_extract_signal(candidates, company_name)
    except Exception as e:
        print(f"[llm] extract_signal fell back to mock due to: {e}")
        return _mock_extract_signal(candidates, company_name)


_CATEGORY_BASE_CONFIDENCE = {
    "manual_context": 0.7, "news": 0.6, "hiring": 0.5, "person_mention": 0.45, "company_overview": 0.3,
}

_CATEGORY_PHRASING = {
    "news": "I saw the recent news that {snippet}",
    "hiring": "Noticed the team's expanding -- {snippet}",
    # "person_mention": "Came across a mention of you -- {snippet}",
    "company_overview": "Looked into what {company} does -- {snippet}",
    "manual_context": "{snippet}",
}


def _lowercase_first(s: str, company_name: str = "") -> str:
    if not s:
        return s
    first_word = s.split(" ", 1)[0].strip(".,")
    # don't lowercase if the first word IS the company name (proper noun)
    if company_name and difflib.SequenceMatcher(
        None, first_word.lower(), company_name.lower().split(".")[0]
    ).ratio() > 0.8:
        return s
    return s[:1].lower() + s[1:]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_DANGLING_TRAILERS = re.compile(
    r"[\s,;:\-–—]+$|(?:\s+(?:and|or|with|for|to|of|in|on|at|by|the|a|an))$",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list:
    """Lightweight regex sentence splitter -- good enough for short scraped
    snippets (not literary prose), and has zero extra dependency or
    runtime data download, unlike spaCy/NLTK. Given two of this project's
    "extra" sources already failed in practice from network flakiness
    (GDELT timeouts, Reddit 403s), adding a dependency that needs its own
    network fetch (NLTK's punkt data) or a much heavier install (spaCy +
    a language model) for what's fundamentally a mechanical cleanup step
    isn't worth the added fragility."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _truncate_at_word(s: str, limit: int = 140) -> str:
    """
    Truncates to a coherent stopping point, in priority order:
      1. Fit as many WHOLE sentences as possible under `limit`.
      2. If even the first sentence is too long, cut at the last clause
         boundary (comma) before `limit` rather than an arbitrary word --
         avoids a dangling trailing fragment like '...with communities,'.
      3. Falls back to the old last-space cut only if neither applies.
    In every case, strips any trailing dangling connector/punctuation
    ('...and', '...with', trailing comma/dash) so the result always reads
    as a complete thought rather than a cut-off list item.
    """
    s = (s or "").strip()
    if not s:
        return s
    if len(s) <= limit:
        return _DANGLING_TRAILERS.sub("", s).rstrip(".")

    sentences = _split_sentences(s)
    if sentences:
        kept = ""
        for sent in sentences:
            candidate = (kept + " " + sent).strip() if kept else sent
            if len(candidate) > limit:
                break
            kept = candidate
        if kept:
            return _DANGLING_TRAILERS.sub("", kept).rstrip(".")
        # even the first sentence alone exceeds the limit -- fall through
        # to clause-level truncation of just that first sentence
        s = sentences[0]

    cut = s[:limit]
    last_comma = cut.rfind(",")
    last_space = cut.rfind(" ")
    if last_comma > limit * 0.5:
        cut = cut[:last_comma]
    elif last_space > limit * 0.6:
        cut = cut[:last_space]
    return _DANGLING_TRAILERS.sub("", cut.strip()).rstrip(".")


def _mock_extract_signal(candidates: list, company_name: str = ""):
    """Deterministic fallback -- no model required, pipeline still runs."""
    ranked = []
    for c in candidates:
        blob = (c["title"] + " " + c["snippet"]).lower()
        negative_words = ["layoff", "lawsuit", "scandal", "fraud", "sued", "bankrupt"]
        sentiment = "negative" if any(w in blob for w in negative_words) else "neutral"
        confidence = _CATEGORY_BASE_CONFIDENCE.get(c["category"], 0.3)
        raw_snippet = _truncate_at_word(c["snippet"] or c["title"])
        # manual_context has no lead-in phrase (template is bare "{snippet}"),
        # so it's a standalone sentence -- lowercasing its first word would
        # mangle the user's own capitalization (e.g. a name). Only lowercase
        # for categories whose template embeds the snippet mid-sentence.
        if c["category"] == "manual_context":
            snippet_short = raw_snippet
        else:
            snippet_short = _lowercase_first(raw_snippet, company_name)
        template = _CATEGORY_PHRASING.get(c["category"], "{snippet}")
        hook_sentence = template.format(snippet=snippet_short, company=company_name or "the company") + "."
        ranked.append({
            **c,
            "hook_sentence": hook_sentence,
            "confidence": confidence,
            "sentiment": sentiment,
            "reasoning": "mock heuristic (Ollama not available)",
        })
    ranked.sort(key=lambda x: x["confidence"], reverse=True)
    return ranked


# ---------- Draft generation ----------

def generate_draft(prospect_name: str, company_name: str, title: str, hook: dict):
    if not _OLLAMA_AVAILABLE:
        return _mock_draft(prospect_name, company_name, title, hook)

    prompt = f"""Write a short, personalized cold outreach email.

You are writing on behalf of {SENDER['company_name']}, {SENDER['one_liner']}

Recipient: {prospect_name}{f", {title}" if title else ""} at {company_name}
Hook to reference naturally (paraphrase, don't quote verbatim): {hook['hook_sentence']}

Rules:
- Under 120 words
- One clear, low-pressure call to action
- No generic flattery, no exclamation points, no "I hope this finds you well"
- Sounds like a specific, busy human wrote it, not a template
- Reference the hook naturally in your own words, don't copy it verbatim
- Briefly connect why {SENDER['company_name']} is relevant to THIS hook --
  don't just append a generic pitch, tie it to what you referenced

Respond with ONLY a JSON object:
{{"subject": "<subject line>", "body": "<email body>"}}
"""
    try:
        parsed = _call_ollama_json(prompt)
        return {
            "subject": parsed.get("subject", f"Quick thought for {company_name}"),
            "body": parsed.get("body", ""),
        }
    except Exception as e:
        print(f"[llm] generate_draft fell back to mock due to: {e}")
        return _mock_draft(prospect_name, company_name, title, hook)


def _mock_draft(prospect_name: str, company_name: str, title: str, hook: dict):
    first_name = prospect_name.split()[0] if prospect_name else "there"
    body = (
        f"Hi {first_name},\n\n"
        f"{hook['hook_sentence']}\n\n"
        f"I'm reaching out from {SENDER['company_name']} -- {SENDER['one_liner']} "
        f"Given what's going on at {company_name}, timing seemed worth a note.\n\n"
        f"Open to a quick 15-minute call? No deck, just a couple of real questions.\n\n"
        f"Best,\n{SENDER['sender_name']}"
    )
    return {"subject": f"Quick note for {company_name}", "body": body}