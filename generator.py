import json
from datetime import date
from llm_client import call_groq


def generate_shared_context(user_request: str, document_type: str, plan: list) -> dict:
    """
    Generates one consistent set of mock facts (names, dates, budget, etc.)
    up front, so every section and every chart pulls from the same source
    of truth instead of each LLM call inventing its own.
    Returns a plain dict of fact -> value. Falls back to an empty dict if
    the LLM response isn't valid JSON, so a single bad call can't break
    the whole run -- sections just fall back to inventing their own details.
    """
    today = date.today().isoformat()
    prompt = f"""You are preparing shared background facts for a {document_type.replace('_', ' ')} document.

Today's date is {today}.

Original user request: "{user_request}"
Planned sections: {", ".join(plan)}

Invent a single consistent set of mock facts to be reused across every section
of this document (so names, dates, and numbers never contradict each other).
Any dates you invent (start_date, end_date, milestones, deadlines, etc.) must
be plausible future dates relative to today -- never dates in the past.
Return ONLY valid JSON, no other text, e.g.:
{{
  "project_name": "...",
  "project_manager": "...",
  "start_date": "...",
  "end_date": "...",
  "total_budget_usd": 50000,
  "other_key_people": ["...", "..."]
}}

Include only fields that make sense for this document type."""

    try:
        response = call_groq(prompt)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except Exception:
        return {}


def generate_document_content(user_request: str, document_type: str, plan: list, template_text: str, shared_context: dict = None) -> dict:
    """
    For each step in the plan, generate real written content,
    using the retrieved template as a structural/style guide and the
    shared_context facts so sections stay consistent with each other.
    Returns a dict like {"step name": "generated content", ...}
    """
    shared_context = shared_context or {}
    context_block = (
        f"\nUse these shared facts consistently (do not invent different ones):\n{json.dumps(shared_context, indent=2)}\n"
        if shared_context else ""
    )

    sections = {}

    for step in plan:
        prompt = f"""You are writing a section of a {document_type.replace('_', ' ')} document.

Reference structure for this document type:
{template_text}
{context_block}
Original user request: "{user_request}"

Write the content for this specific section: "{step}"

Write 2-4 sentences of realistic, professional business content. Use mock/placeholder details where specific facts aren't provided (dates, names, numbers) -- but stay consistent with the shared facts above if given. Return ONLY the section content, no headers or extra text."""

        try:
            content = call_groq(prompt)
            sections[step] = content.strip()
        except Exception as e:
            sections[step] = f"[Content generation failed for this section: {e}]"

    return sections
