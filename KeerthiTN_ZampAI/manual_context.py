"""
Manual context input -- lets the rep paste text or upload a PDF/DOCX/TXT
with anything they already know about the prospect (a bio, a LinkedIn
export, notes from a call, an intro email). This exists specifically for
the case research turns up little or nothing: instead of the pipeline
either fabricating a hook or refusing outright, a human-supplied fact
becomes a legitimate, highest-priority candidate.

Free/open-source only: pypdf for PDF text extraction, python-docx for
.docx -- both pure-Python, no external services or keys.
"""
import io

MAX_CONTEXT_CHARS = 4000  # keep this bounded -- it goes straight into an LLM prompt


def extract_text_from_upload(file_storage) -> str:
    """
    file_storage: a werkzeug FileStorage object from request.files.
    Returns extracted plain text, or "" if the type is unsupported / parsing fails.
    Never raises -- a bad upload should degrade to "no manual context",
    not break the run.
    """
    if not file_storage or not file_storage.filename:
        return ""
    filename = file_storage.filename.lower()
    data = file_storage.read()
    try:
        if filename.endswith(".pdf"):
            return _extract_pdf(data)
        if filename.endswith(".docx"):
            return _extract_docx(data)
        if filename.endswith(".txt") or filename.endswith(".md"):
            return data.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[manual_context] failed to parse {filename}: {e}")
        return ""
    print(f"[manual_context] unsupported file type: {filename} -- only .pdf, .docx, .txt supported")
    return ""


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def build_manual_context(pasted_text: str, file_storage=None) -> str:
    """Combines pasted text + any uploaded file's extracted text into one
    bounded string, or "" if neither was provided."""
    parts = []
    if pasted_text and pasted_text.strip():
        parts.append(pasted_text.strip())
    file_text = extract_text_from_upload(file_storage) if file_storage else ""
    if file_text.strip():
        parts.append(file_text.strip())
    combined = "\n\n".join(parts)
    return combined[:MAX_CONTEXT_CHARS]
