"""
PS-3: Personalized outreach -- from prospect to draft. (v2)

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""
import json
import sqlite3
from datetime import datetime, timezone

from flask import Flask, Response, request, jsonify, render_template, stream_with_context

import research
import llm
import manual_context as manual_context_mod

app = Flask(__name__)
DB_PATH = "runs.db"

STALE_DAYS = 180
MIN_CONFIDENCE = 0.35
MIN_IDENTIFICATION_STRENGTH = 0.15


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prospect_name TEXT,
            company_name TEXT,
            title TEXT,
            status TEXT,
            hook_category TEXT,
            hook TEXT,
            confidence REAL,
            id_label TEXT,
            flags TEXT,
            subject TEXT,
            body TEXT,
            source_url TEXT,
            company_snapshot TEXT
        )
    """)
    # Auto-migrate older DBs (e.g. from v1) that are missing newer columns,
    # instead of crashing mid-pipeline the way a bare CREATE TABLE IF NOT
    # EXISTS would.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    needed_cols = {
        "hook_category": "TEXT", "id_label": "TEXT", "company_snapshot": "TEXT",
    }
    for col, coltype in needed_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()


def save_run(record: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO runs (timestamp, prospect_name, company_name, title, status,
                           hook_category, hook, confidence, id_label, flags,
                           subject, body, source_url, company_snapshot)
        VALUES (:timestamp, :prospect_name, :company_name, :title, :status,
                :hook_category, :hook, :confidence, :id_label, :flags,
                :subject, :body, :source_url, :company_snapshot)
    """, record)
    conn.commit()
    conn.close()


def _looks_stale(date_str: str) -> bool:
    if not date_str:
        return False
    try:
        d = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - d).days
        return age_days > STALE_DAYS
    except Exception:
        return False


def run_pipeline(prospect_name, company_name, title, manual_context=""):
    """Generator yielding one JSON event per stage -- powers the live run view.

    manual_context: optional free-text the rep supplies directly (pasted,
    or extracted from an uploaded PDF/DOCX) -- e.g. LinkedIn bio, notes
    from a call, an intro email. Becomes a highest-priority candidate hook,
    and specifically lets a run survive when web research alone would
    otherwise come back with nothing usable."""

    def emit(stage, status, detail=None):
        return json.dumps({"stage": stage, "status": status, "detail": detail}) + "\n"

    flags = []
    try:
        yield emit("input", "done", f"prospect={prospect_name!r} company={company_name!r}"
                   + (" manual_context=provided" if manual_context else ""))

        # ---- Stage: research (4 categories) ----
        yield emit("research", "start", "searching company overview, news, hiring, person mentions")
        bundle = research.full_research(prospect_name, company_name)
        yield emit("research", "done", {
            "company_website": bundle.get("company_website"),
            "news_found": len(bundle.get("news", [])),
            "hiring_signals_found": len(bundle.get("hiring_signals", [])),
            "person_mentions_found": len(bundle.get("person_mentions", [])),
            "has_company_overview": bool(bundle.get("about_us")),
        })

        # ---- Stage: employment status check (hard stop, not a rankable hook) ----
        # A negative signal ABOUT THE COMPANY (lawsuit, bad PR) is something
        # to avoid referencing -- skip it, pick another hook. A negative
        # signal ABOUT THE PROSPECT'S EMPLOYMENT (laid off, departed) is
        # different in kind: it invalidates the premise of the whole email
        # ("here's what's happening at {company}" to someone who no longer
        # works there). That can't be fixed by substituting a safer hook --
        # it has to stop the pipeline and go to a human.
        yield emit("employment_check", "start", "checking whether prospect still appears to work at company")
        departure_signal = research.check_employment_status(prospect_name, bundle.get("person_mentions", []))
        if departure_signal:
            yield emit("employment_check", "flag",
                        f"found signal prospect may no longer be at {company_name}: "
                        f"{departure_signal['title'][:100]}")
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prospect_name": prospect_name, "company_name": company_name, "title": title,
                "status": "prospect_may_have_departed", "hook_category": "person_mention",
                "hook": departure_signal["title"], "confidence": 0.0,
                "id_label": research.identification_label(
                    research.identification_strength(prospect_name, company_name, bundle)),
                "flags": "prospect_may_have_departed",
                "subject": "", "body": "", "source_url": departure_signal.get("url", ""),
                "company_snapshot": "",
            }
            save_run(record)
            yield emit("done", "prospect_may_have_departed",
                        f"Found a signal that {prospect_name} may no longer work at {company_name} "
                        f"({departure_signal['title'][:100]}). Halting instead of drafting an email "
                        f"premised on them still being there -- needs human verification first.")
            return
        yield emit("employment_check", "done", "no departure signal found")

        # ---- Stage: name-evidence check (hard stop -- catches fabricated names) ----
        # identification_strength() blends company-confirmed + name-matched
        # into one 0-1 score, and gives 0.5 just for the company being real.
        # That means a made-up name paired with a real company (e.g. a
        # fabricated name at "Oracle") can clear MIN_IDENTIFICATION_STRENGTH
        # on company evidence alone, with zero evidence anyone by that name
        # exists there. Checked here as its own explicit gate, separate from
        # the blended score, so it can't be diluted by unrelated company signal.
        yield emit("name_evidence_check", "start", "checking that at least one source actually names the prospect")
        if not research.has_name_evidence(prospect_name, bundle) and not manual_context:
            yield emit("name_evidence_check", "flag",
                        f"no source found that actually names '{prospect_name}' -- "
                        f"cannot confirm this person exists at {company_name}")
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prospect_name": prospect_name, "company_name": company_name, "title": title,
                "status": "unverified_person", "hook_category": "", "hook": "", "confidence": 0.0,
                "id_label": "Weak match",
                "flags": "no_name_evidence",
                "subject": "", "body": "", "source_url": "",
                "company_snapshot": "",
            }
            save_run(record)
            yield emit("done", "unverified_person",
                        f"No source names '{prospect_name}' specifically -- only the company "
                        f"({company_name}) checked out. Halting instead of drafting a personalized "
                        f"email to someone we can't confirm exists. Needs manual verification "
                        f"(e.g. a LinkedIn URL) before retrying, or paste in what you already know.")
            return
        elif manual_context and not research.has_name_evidence(prospect_name, bundle):
            yield emit("name_evidence_check", "info",
                        "no web source names the prospect, but manual context was provided -- proceeding on that")
        else:
            yield emit("name_evidence_check", "done", "at least one source names the prospect")

        id_score = research.identification_strength(prospect_name, company_name, bundle)
        id_label = research.identification_label(id_score)
        if id_score < MIN_IDENTIFICATION_STRENGTH:
            flags.append("weak_identification")

        # ---- Stage: prospect summary (deterministic, fast, inspectable) ----
        yield emit("prospect_summary", "start", "assembling structured summary")
        summary = llm.build_prospect_summary(company_name, bundle, id_score, id_label)
        yield emit("prospect_summary", "done", summary)

        # ---- Stage: signal extraction ----
        yield emit("signal_extraction", "start", "ranking candidate hooks across all categories")
        candidates = research.flatten_candidates(bundle)
        if manual_context:
            # highest-priority candidate: a human explicitly supplied this,
            # so it doesn't need the same relevance/garbled-text filtering
            # scraped sources go through. This is also what lets a run
            # survive when web research alone comes back with nothing.
            candidates.insert(0, {
                "category": "manual_context",
                "title": f"Context provided for {prospect_name}",
                "snippet": manual_context,
                "url": "", "date": "",
            })
            yield emit("signal_extraction", "info", "included manually provided context as a top-priority candidate")
        ranked = llm.extract_signal(prospect_name, company_name, candidates)
        yield emit("signal_extraction", "done", f"{len(ranked)} ranked candidates")

        # ---- Stage: business rules (judgment layer, kept explicit/out of the LLM) ----
        yield emit("business_rules", "start", "applying policy filters")
        chosen = None
        for c in ranked:
            if c["sentiment"] == "negative":
                yield emit("business_rules", "info",
                            f"discarded [{c['category']}] hook (negative sentiment): {c['title'][:80]}")
                continue
            if c["confidence"] < MIN_CONFIDENCE:
                yield emit("business_rules", "info",
                            f"discarded [{c['category']}] hook (low confidence {c['confidence']:.2f}): {c['title'][:80]}")
                continue
            chosen = c
            break

        if chosen is None:
            yield emit("business_rules", "done", "no hook survived policy filters")
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prospect_name": prospect_name, "company_name": company_name, "title": title,
                "status": "insufficient_signal", "hook_category": "", "hook": "", "confidence": 0.0,
                "id_label": id_label,
                "flags": ",".join(flags) if flags else "no_usable_signal",
                "subject": "", "body": "", "source_url": "",
                "company_snapshot": summary["company_snapshot"],
            }
            save_run(record)
            yield emit("done", "insufficient_signal",
                        "No usable, on-policy signal found. Flagged for manual research "
                        "instead of fabricating a draft.")
            return

        if _looks_stale(chosen.get("date", "")):
            flags.append("stale_signal")
            yield emit("business_rules", "flag", f"chosen hook is >{STALE_DAYS} days old -- marking low-confidence")

        yield emit("business_rules", "done", f"selected [{chosen['category']}] hook: {chosen['title'][:100]}")

        # ---- Stage: draft generation ----
        yield emit("draft_generation", "start", "writing personalized draft")
        draft = llm.generate_draft(prospect_name, company_name, title, chosen)
        yield emit("draft_generation", "done", draft["subject"])

        status = "ready_for_review"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prospect_name": prospect_name, "company_name": company_name, "title": title,
            "status": status, "hook_category": chosen["category"], "hook": chosen["hook_sentence"],
            "confidence": chosen["confidence"], "id_label": id_label,
            "flags": ",".join(flags) if flags else "",
            "subject": draft["subject"], "body": draft["body"], "source_url": chosen.get("url", ""),
            "company_snapshot": summary["company_snapshot"],
        }
        save_run(record)

        yield emit("output", "done", {
            "status": status,
            "subject": draft["subject"],
            "body": draft["body"],
            "hook_category": chosen["category"],
            "hook": chosen["hook_sentence"],
            "confidence": round(chosen["confidence"], 2),
            "id_label": id_label,
            "flags": flags,
            "source_url": chosen.get("url", ""),
            "summary": summary,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            save_run({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prospect_name": prospect_name, "company_name": company_name, "title": title,
                "status": "error", "hook_category": "", "hook": "", "confidence": 0.0,
                "id_label": "", "flags": "pipeline_error",
                "subject": "", "body": "", "source_url": "", "company_snapshot": "",
            })
        except Exception:
            pass
        yield emit("error", "failed", f"{type(e).__name__}: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    # The form now supports an optional file upload, so the request comes
    # in as multipart/form-data rather than JSON. Still accept plain JSON
    # too (e.g. for programmatic/API-only callers with no file to attach).
    if request.content_type and "multipart/form-data" in request.content_type:
        prospect_name = (request.form.get("prospect_name") or "").strip()
        company_name = (request.form.get("company_name") or "").strip()
        title = (request.form.get("title") or "").strip()
        pasted_context = request.form.get("manual_context") or ""
        uploaded_file = request.files.get("context_file")
        context = manual_context_mod.build_manual_context(pasted_context, uploaded_file)
    else:
        data = request.get_json(force=True, silent=True) or {}
        prospect_name = (data.get("prospect_name") or "").strip()
        company_name = (data.get("company_name") or "").strip()
        title = (data.get("title") or "").strip()
        context = (data.get("manual_context") or "").strip()

    if not prospect_name or not company_name:
        return jsonify({"error": "prospect_name and company_name are required"}), 400

    return Response(
        stream_with_context(run_pipeline(prospect_name, company_name, title, context)),
        mimetype="application/x-ndjson",
    )


@app.route("/api/history")
def api_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history/<int:run_id>", methods=["DELETE"])
def api_delete_history(run_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": run_id})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)