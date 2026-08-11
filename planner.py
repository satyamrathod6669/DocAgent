import json
from llm_client import call_groq

VALID_DOC_TYPES = {"project_plan", "meeting_minutes", "business_proposal", "sop", "technical_design"}

PROMPT_TEMPLATE = """You are a planning assistant. Given a user's request, determine:
1. The type of business document needed (choose one: project_plan, meeting_minutes, business_proposal, sop, technical_design)
2. A list of 4-6 steps needed to create this document
3. If the request is vague or missing information, state your assumption

Return ONLY valid JSON in this exact format, no other text:
{{
  "document_type": "...",
  "assumption": "...",
  "plan": ["step 1", "step 2", "..."]
}}

User request: "{request}"
"""


def _parse_plan_response(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    data = json.loads(cleaned.strip())

    # Validate shape so downstream steps can rely on it
    if not isinstance(data, dict):
        raise ValueError("Plan response was not a JSON object")
    if "document_type" not in data or "plan" not in data:
        raise ValueError("Plan response missing required fields")
    if not isinstance(data["plan"], list) or len(data["plan"]) == 0:
        raise ValueError("Plan response 'plan' field must be a non-empty list")
    data.setdefault("assumption", "")

    return data


def create_plan(user_request: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(request=user_request)

    last_error = None
    for attempt in range(2):  # one retry if the LLM returns malformed JSON
        try:
            response = call_groq(prompt)
            return _parse_plan_response(response)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            last_error = e

    raise ValueError(f"Planner failed to produce a valid plan after retrying: {last_error}")
