"""
Research module (v5).
Pulls signal from reliable free/open sources only, no API keys anywhere:
  1. company overview       -- free web search + lightweight scraping (requests/bs4)
  2. recent news             -- DuckDuckGo news search
  3. hiring signals           -- DuckDuckGo text search for careers/hiring mentions
  4. person mentions          -- DuckDuckGo text search for "{person}" "{company}"
  5. Wikipedia summary        -- Wikipedia REST API (company verification fallback)
  6. Hacker News mentions     -- HN Algolia Search API
"""
import re
import difflib

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

SEARCH_TIMEOUT = 8
FETCH_TIMEOUT = 6
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OutreachResearchBot/1.0)"}

_SUFFIXES = [".ai", ".com", ".io", ".co", ".net", ".org", ".in",
             " inc", " llc", " ltd", " corp", " corporation",
             " pvt", " private limited", " technologies", " technology"]



def normalize_company_name(company_name: str) -> str:
    """'Zamp.ai' -> 'zamp' -- strips domain/legal suffixes so identification
    matching isn't fooled by a suffix that rarely appears in prose mentions."""
    name = company_name.strip().lower()
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def _fuzzy_word_hit(token: str, blob_words: list, threshold: float = 0.85) -> bool:
    if not token:
        return False
    return any(difflib.SequenceMatcher(None, token, w).ratio() > threshold for w in blob_words)


def _company_mentioned(company_name: str, blob: str) -> bool:
    """Fuzzy, suffix-aware check for whether a company is actually referenced."""
    norm = normalize_company_name(company_name)
    if not norm:
        return False
    blob_l = blob.lower()
    if norm in blob_l:  # cheap exact-substring fast path
        return True
    words = re.findall(r"[a-zA-Z]+", blob_l)
    core_token = norm.split()[0] if norm.split() else norm
    return _fuzzy_word_hit(core_token, words)


def _safe_search_news(query, max_results=5):
    try:
        with DDGS(timeout=SEARCH_TIMEOUT) as ddgs:
            return list(ddgs.news(query, max_results=max_results))
    except Exception as e:
        print(f"[research] news search failed: {e}")
        return []


def _safe_search_text(query, max_results=5):
    try:
        with DDGS(timeout=SEARCH_TIMEOUT) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[research] text search failed: {e}")
        return []


def _clean_scraped_text(text: str) -> str:
    """
    Raw scraped/DDG snippet text sometimes glues fragments together with no
    separator -- e.g. 'skoob.aiAt Skoob, we craft...' or 'topic11-50
    Employees'. Insert a space at the two most common collision points
    (lowercase->uppercase, letter->digit) and collapse whitespace, so a
    downstream hook sentence doesn't inherit the garbling verbatim.
    """
    if not text:
        return text
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)  # wordEndWord -> word EndWord, but leaves PhD/AI/GPU intact
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)       # topic11 -> topic 11
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)       # 11-50Employees -> 11-50 Employees
    text = re.sub(r"\s+", " ", text).strip()
    # DDG sometimes prefixes a snippet with the bare domain as page
    # boilerplate ("skoob.ai At Skoob, we craft...") -- strip a leading
    # standalone domain-shaped token, it's not part of the actual sentence.
    text = re.sub(r"^[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\s+", "", text)
    # DDG also frequently prefixes listing snippets with a relative
    # timestamp -- "3 days ago · You'll work closely..." -- that's UI
    # metadata from the source page, not prose, and reads as a stale/cliche
    # opener if it leaks into a hook sentence. Strip it.
    text = re.sub(
        r"^\d+\s*(?:hour|hours|day|days|week|weeks|month|months|year|years)\s+ago\s*[·•\-–]\s*",
        "", text, flags=re.IGNORECASE,
    )
    return text


def _looks_garbled(text: str) -> bool:
    """
    Even after cleaning, some scraped text is just not a coherent sentence
    -- rejected as a hook candidate rather than shipped into an email.
    Heuristics: too few spaces relative to length (wall of concatenated
    words), or an unreasonably long single "word" (URL slug, id string).
    """
    if not text or len(text) < 8:
        return True
    words = text.split()
    if not words:
        return True
    avg_word_len = len(text) / len(words)
    longest_word = max(len(w) for w in words)
    return avg_word_len > 14 or longest_word > 30


def _dedupe_by_url(items, url_key="url"):
    seen, out = set(), []
    for item in items:
        u = item.get(url_key, "")
        if u and u in seen:
            continue
        seen.add(u)
        out.append(item)
    return out


def _clean_title(title: str, max_len: int = 90) -> str:
    """
    DuckDuckGo's scraper occasionally returns a malformed 'title' that's
    actually several unrelated result titles mashed together (e.g.
    '...LinkedIn...RocketReach...Tracxn...Inc42...'). Detect that pattern
    and cut it down to just the first real segment instead of showing the
    garbled blob.
    """
    if not title:
        return title
    if len(title) <= max_len and title.count("|") <= 2:
        return title
    # split on common separators and keep the first coherent chunk
    for sep in (" | ", " - "):
        if sep in title:
            first = title.split(sep)[0].strip()
            if 3 < len(first) <= max_len:
                return first
    return title[:max_len].rsplit(" ", 1)[0] + "..."
    seen, out = set(), []
    for item in items:
        u = item.get(url_key, "")
        if u and u in seen:
            continue
        seen.add(u)
        out.append(item)
    return out


# ---------- Category 1: company overview ----------

def _looks_like_domain(company_name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}$", company_name.strip()))


def _url_domain_matches(url: str, expected_domain: str) -> bool:
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower().replace("www.", "")
    return netloc == expected_domain.lower()


def find_company_website(company_name):
    """
    Resolves the company's real website. If the person typed something that
    already looks like a literal domain (e.g. 'Skoob.ai'), that domain is
    tried FIRST and verified -- otherwise a generic name search can silently
    return an unrelated company that happens to share the same name on a
    different TLD (e.g. 'skoob.com', a London bookshop, instead of the
    intended 'skoob.ai').
    """
    stripped = company_name.strip()
    if _looks_like_domain(stripped):
        for candidate in (f"https://{stripped}", f"https://www.{stripped}"):
            try:
                resp = requests.get(candidate, headers=HEADERS, timeout=FETCH_TIMEOUT, allow_redirects=True)
                if resp.status_code < 400:
                    return candidate
            except Exception:
                continue

    norm = normalize_company_name(company_name)
    # search with the ORIGINAL name first (keeps TLD/disambiguating signal),
    # normalized name only as a fallback
    results = _safe_search_text(f'"{company_name}" official website', max_results=3)
    results += _safe_search_text(f"{norm} official website", max_results=3)
    for r in results:
        url = r.get("href", "")
        if url and "linkedin.com" not in url and "wikipedia.org" not in url:
            return url
    return results[0].get("href") if results else None


def fetch_about_us(url, company_name=""):
    if not url:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        # requests sometimes guesses the wrong encoding for a page (falls back
        # to latin-1), which turns real characters like em-dashes/ellipses
        # into mojibake ("â€¦"). apparent_encoding sniffs it from the actual
        # bytes and is far more reliable.
        if resp.encoding is None or resp.encoding.lower() in ("iso-8859-1", "latin-1"):
            resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        meta = soup.find("meta", attrs={"name": "description"})
        desc = meta["content"].strip() if meta and meta.get("content") else ""
        if not desc:
            p = soup.find("p")
            desc = p.get_text(strip=True) if p else ""
        title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
        # Sanity check: if we know the intended company name, make sure this
        # page is actually plausibly about it (page title/description mentions
        # it) before treating it as authoritative -- prevents a same-named but
        # unrelated company's site from silently becoming the "company snapshot"
        if company_name and not _company_mentioned(company_name, f"{title} {desc}"):
            # still worth returning (it IS the domain that was resolved), but
            # flagged as unverified so callers can decide whether to trust it
            return {"url": url, "title": title, "description": desc[:500], "verified": False}
        return {"url": url, "title": title, "description": desc[:500], "verified": True}
    except Exception as e:
        print(f"[research] website fetch failed: {e}")
        return None


# ---------- Category 2: news -- two phrasings, combined & deduped ----------

_AGGREGATOR_TITLE_PATTERNS = [
    "latest news, analysis, funding", "company profile", "overview, competitors",
]
_JOB_BOARD_AD_PATTERNS = [
    "take your first step towards your dream job", "apply now in", "get placed",
    "with jobaaj", "top companies hiring", "explore jobs at top",
]
_PROFILE_FIELD_LABELS = [
    "experience:", "education:", "location:", "connections", "followers",
    "endorsements", "skills:", "about:", "headline:",
]


def _is_profile_boilerplate(text: str) -> bool:
    """
    Catches raw LinkedIn/directory profile metadata masquerading as a
    "mention" of the person -- e.g. 'Experience: Skoob.ai · Education: IIT
    Kanpur · Location: Bengaluru · 500+ connections on LinkedIn'. This is
    real information (it confirms the person exists and roughly who they
    are), but it was never a sentence -- it's a list of field:value pairs
    scraped off a profile page. Templating it directly produces exactly
    the kind of email nobody wants to receive: 'Came across a mention of
    you -- experience: X, education: Y...' reads like a broken bot, not a
    human who did research.

    Heuristic: 2+ recognizable profile-field labels present, OR the
    "N+ connections on LinkedIn" phrase, OR 2+ mid-string "·" separators
    (LinkedIn's own field-separator character in search snippets).
    """
    blob = text.lower()
    label_hits = sum(1 for label in _PROFILE_FIELD_LABELS if label in blob)
    if label_hits >= 2:
        return True
    if re.search(r"\d+\+?\s*connections?\s+on\s+linkedin", blob):
        return True
    if blob.count("·") >= 2:
        return True
    return False


def _is_aggregator_page(item: dict) -> bool:
    title_l = item.get("title", "").lower()
    return any(p in title_l for p in _AGGREGATOR_TITLE_PATTERNS)


def _is_job_board_ad(item: dict) -> bool:
    """
    Catches generic job-board marketing copy (e.g. a Jobaaj/Indeed listing
    page ad like 'Apply now... take your first step towards your Dream Job')
    that mentions the company incidentally but isn't an actual hiring signal
    FROM the company itself.
    """
    blob = f"{item.get('title','')} {item.get('snippet','')}".lower()
    return any(p in blob for p in _JOB_BOARD_AD_PATTERNS)


def gather_news(company_name, max_results=5):
    norm = normalize_company_name(company_name)
    raw = _safe_search_news(f'"{company_name}" news funding launch announcement', max_results=max_results)
    raw += _safe_search_news(f"{norm} news announcement", max_results=max_results)
    items = [{
        "title": _clean_title(r.get("title", "")), "snippet": r.get("body", ""),
        "url": r.get("url", ""), "date": r.get("date", ""), "source": r.get("source", ""),
    } for r in raw]
    items = _dedupe_by_url(items)
    items = [i for i in items if _company_mentioned(company_name, i["title"] + " " + i["snippet"])]
    # drop generic aggregator/directory profile pages -- they aren't real
    # news events and their "date" is unreliable (causes false stale flags)
    return [i for i in items if not _is_aggregator_page(i)]


# ---------- Category 3: hiring signals -- wider net ----------

def fetch_hiring_signals(company_name, max_results=5):
    norm = normalize_company_name(company_name)
    raw = _safe_search_text(
        f'"{company_name}" hiring OR careers OR "we are hiring" OR "join our team"',
        max_results=max_results,
    )
    raw += _safe_search_text(f"{norm} careers open roles", max_results=max_results)
    items = [{
        "title": _clean_title(r.get("title", "")), "snippet": r.get("body", ""), "url": r.get("href", ""),
    } for r in raw]
    items = _dedupe_by_url(items)
    items = [i for i in items if _company_mentioned(company_name, i["title"] + " " + i["snippet"])]
    # drop generic job-board marketing copy (e.g. a Jobaaj ad that mentions
    # the company incidentally) -- not a real signal FROM the company
    return [i for i in items if not _is_job_board_ad(i)]


# ---------- Category 4: person mentions -- wider net ----------

def _person_mentioned(prospect_name: str, blob: str) -> bool:
    """
    Fuzzy check that a result actually references THIS person by name --
    not just something DuckDuckGo loosely associated with the query.
    Mirrors _company_mentioned(). Requires the last name (the more
    distinguishing token) to appear, fuzzy-matched against blob words.
    """
    parts = [p for p in re.findall(r"[a-zA-Z]+", prospect_name.lower()) if len(p) > 1]
    if not parts:
        return False
    last = parts[-1]
    blob_words = re.findall(r"[a-zA-Z]+", blob.lower())
    return _fuzzy_word_hit(last, blob_words)


_DEPARTURE_PATTERNS = [
    "laid off", "layoff", "layoffs", "let go", "no longer works",
    "no longer with", "former employee", "formerly at", "formerly worked",
    "left the company", "departed", "was terminated", "was fired",
    "affected by layoffs", "impacted by layoffs",
]


def gather_person_mentions(prospect_name, company_name, max_results=6):
    norm = normalize_company_name(company_name)
    raw = _safe_search_text(f'"{prospect_name}" "{company_name}"', max_results=max_results)
    raw += _safe_search_text(f'"{prospect_name}" {norm} linkedin', max_results=max_results)
    # widen the net further: search departure language explicitly, since a
    # plain "name + company" query often won't surface a layoff story if the
    # article doesn't happen to name-check the company in the same breath
    raw += _safe_search_text(f'"{prospect_name}" layoff OR laid off OR "no longer"', max_results=max_results)
    items = [{
        "title": _clean_title(r.get("title", "")), "snippet": r.get("body", ""), "url": r.get("href", ""),
    } for r in raw]
    items = _dedupe_by_url(items)
    # relevance filter (mirrors company-mention filtering elsewhere): only
    # keep results that actually reference this person by name -- a result
    # that mentions the company but not the person is not a "person mention"
    return [i for i in items if _person_mentioned(prospect_name, i["title"] + " " + i["snippet"])]


def check_employment_status(prospect_name, person_mentions):
    """
    Scans NAME-VERIFIED person mentions for language indicating the
    prospect has left the company (layoff, departure, termination).
    Returns the first matching mention dict, or None.

    This is deliberately separate from generic sentiment scoring: a
    "negative" story about the prospect's employment status isn't a hook to
    rank against other hooks -- it's a signal that the entire premise of a
    "here's what's happening at {company}" email is wrong, and needs a
    human, not a fallback hook.
    """
    for m in person_mentions:
        blob = f"{m['title']} {m['snippet']}".lower()
        if any(p in blob for p in _DEPARTURE_PATTERNS):
            return m
    return None


# ---------- Category 5-8: additional free/no-key sources ----------

def fetch_wikipedia_summary(company_name):
    """
    Wikipedia REST API summary endpoint -- free, no key. Useful as a second
    opinion on company identity (notable companies almost always have a
    page) and occasionally surfaces exec/person pages too.
    """
    norm = normalize_company_name(company_name).title()
    try:
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{norm.replace(' ', '_')}",
            headers=HEADERS, timeout=FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract", "")
        if not extract:
            return None
        return {
            "title": data.get("title", norm),
            "snippet": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:
        print(f"[research] wikipedia fetch failed: {e}")
        return None


def search_hackernews(query, max_results=5):
    """Hacker News Algolia Search API -- free, no key."""
    try:
        resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story,comment", "hitsPerPage": max_results},
            headers=HEADERS, timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        return [{
            "title": _clean_title(h.get("title") or h.get("story_title") or query),
            "snippet": (h.get("comment_text") or h.get("story_text") or "")[:400],
            "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID','')}",
            "date": h.get("created_at", ""),
        } for h in hits]
    except Exception as e:
        print(f"[research] hackernews search failed: {e}")
        return []







def full_research(prospect_name, company_name):
    """Runs all sources (each with a wider multi-query net where relevant)
    and returns one combined, deduped bundle -- more raw candidates for the
    ranking step to choose the best from, and more chances for a REAL
    person to actually surface verifiable evidence.

    Only sources that are reliably reachable without a key are wired in
    here. GDELT (frequently times out) and Reddit's search.json (blocks
    server-side requests with 403s in practice) were tried and dropped --
    a source that fails most of the time adds noise and latency to every
    run without adding real signal. Simple and working beats wide and flaky."""
    website = find_company_website(company_name)
    about = fetch_about_us(website, company_name)

    # Wikipedia as a second opinion on company identity -- if the primary
    # about_us page didn't verify (or wasn't found), Wikipedia often will.
    if not about or not about.get("verified", True):
        wiki = fetch_wikipedia_summary(company_name)
        if wiki and _company_mentioned(company_name, wiki["title"] + " " + wiki["snippet"]):
            about = {"url": wiki["url"], "title": wiki["title"],
                      "description": wiki["snippet"][:500], "verified": True}

    news = gather_news(company_name)
    hiring = fetch_hiring_signals(company_name)

    mentions = gather_person_mentions(prospect_name, company_name)
    # HN Algolia as an additional person-mention source -- same relevance
    # filter applies (must actually name the person), so a fabricated name
    # gains nothing from this wider net; a real, findable person does.
    extra_mentions = search_hackernews(f'"{prospect_name}" {company_name}')
    extra_mentions = [m for m in extra_mentions
                       if _person_mentioned(prospect_name, m["title"] + " " + m["snippet"])]
    mentions = _dedupe_by_url(mentions + extra_mentions)

    return {
        "company_website": website,
        "about_us": about,
        "news": news,
        "hiring_signals": hiring,
        "person_mentions": mentions,
    }


def flatten_candidates(bundle):
    """Flattens the category bundles into one list of hook candidates,
    tagged with category so the UI can show WHERE a hook came from.
    Snippets are cleaned (see _clean_scraped_text) and candidates whose
    text still reads as garbled after cleaning are dropped here -- this is
    the one place every source funnels through, so it's the right
    chokepoint for the quality gate rather than repeating it per-source."""
    candidates = []

    def _add(category, title, snippet, url, date=""):
        snippet = _clean_scraped_text(snippet)
        title = _clean_scraped_text(title)
        if _looks_garbled(snippet) and _looks_garbled(title):
            return
        candidates.append({"category": category, "title": title, "snippet": snippet,
                            "url": url, "date": date})

    for n in bundle.get("news", []):
        _add("news", n["title"], n["snippet"], n["url"], n.get("date", ""))
    for h in bundle.get("hiring_signals", []):
        _add("hiring", h["title"], h["snippet"], h["url"])
    for m in bundle.get("person_mentions", []):
        # profile metadata (LinkedIn field dumps) confirms the person
        # exists -- valuable for identification_strength/has_name_evidence,
        # both of which read bundle["person_mentions"] directly and are
        # unaffected by this filter -- but it was never a sentence, so it
        # doesn't get to become a hook candidate. See _is_profile_boilerplate.
        if _is_profile_boilerplate(f"{m['title']} {m['snippet']}"):
            continue
        _add("person_mention", m["title"], m["snippet"], m["url"])
    about = bundle.get("about_us")
    if about and about.get("description") and about.get("verified", True):
        _add("company_overview", about.get("title", ""), about["description"], about["url"])
    return candidates


def has_name_evidence(prospect_name: str, bundle: dict) -> bool:
    """
    True only if at least one source actually names the prospect. This is
    checked SEPARATELY from identification_strength()'s blended 0-1 score,
    because that blend gives 0.5 credit just for the company being real --
    which a fabricated name paired with a real company (e.g. 'Rina Kapoor'
    at 'Oracle') satisfies trivially. A fabricated-name case should fail on
    the ONE thing it actually can't fake: someone, somewhere, naming them.
    """
    name_parts = [p for p in prospect_name.lower().split() if len(p) > 2]
    if not name_parts:
        return False
    for m in bundle.get("person_mentions", []):
        blob = f"{m['title']} {m['snippet']}".lower()
        words = re.findall(r"[a-zA-Z]+", blob)
        if any(_fuzzy_word_hit(part, words) for part in name_parts):
            return True
    return False


# ---------- Identification confidence: combined evidence across ALL sources ----------

def identification_strength(prospect_name: str, company_name: str, bundle: dict):
    """
    How confidently did we find THIS person at THIS company?
    Combines evidence across every research category (not just the single
    best person-mention snippet), and uses normalized/fuzzy company matching
    so 'Zamp.ai' correctly matches a mention that just says 'Zamp'.
    Returns 0-1.
    """
    all_candidates = flatten_candidates(bundle)

    about = bundle.get("about_us")
    about_confirms = bool(about) and about.get("verified", True) and bool(about.get("description"))
    company_confirmed = about_confirms or any(
        _company_mentioned(company_name, f"{c['title']} {c['snippet']}")
        for c in all_candidates
    )

    name_parts = [p for p in prospect_name.lower().split() if len(p) > 2]
    matched_parts = set()
    for m in bundle.get("person_mentions", []):
        blob = f"{m['title']} {m['snippet']}".lower()
        words = re.findall(r"[a-zA-Z]+", blob)
        for part in name_parts:
            if _fuzzy_word_hit(part, words):
                matched_parts.add(part)
    name_score = len(matched_parts) / max(len(name_parts), 1)

    score = (0.5 * company_confirmed) + (0.5 * name_score)
    return round(score, 2)


def identification_label(score: float) -> str:
    if score >= 0.7:
        return "Strong match"
    if score >= 0.4:
        return "Partial match"
    return "Weak match"