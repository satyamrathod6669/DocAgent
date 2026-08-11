import json
import os
import matplotlib
from datetime import date
matplotlib.use("Agg")  # no GUI backend needed, just saving files
import matplotlib.pyplot as plt

from llm_client import call_groq


def _ask_for_json(prompt: str) -> dict:
    """Calls the LLM and parses its response as JSON, cleaning markdown fences if present."""
    response = call_groq(prompt)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def get_budget_breakdown(user_request: str, shared_context: dict = None) -> dict:
    """Asks the LLM for a plausible budget breakdown as category -> amount (USD)."""
    context_hint = ""
    if shared_context and shared_context.get("total_budget_usd"):
        context_hint = (
            f"\nThe total budget should sum to approximately "
            f"{shared_context['total_budget_usd']} USD, matching the figure already "
            f"used elsewhere in the document."
        )

    prompt = f"""Based on this request: "{user_request}"
{context_hint}
Generate a plausible budget breakdown for the project. Return ONLY valid JSON,
no other text, in this exact format:
{{
  "Category Name 1": 12000,
  "Category Name 2": 8000
}}

Use 4-6 realistic categories (e.g. Personnel, Design, Development, Testing,
Marketing, Contingency). Amounts should be whole numbers in USD, and should
roughly sum to a reasonable total project budget."""
    return _ask_for_json(prompt)


def get_timeline_milestones(user_request: str, shared_context: dict = None) -> list:
    """Asks the LLM for a plausible timeline as a list of {milestone, week} objects."""
    today = date.today().isoformat()
    context_hint = ""
    if shared_context and (shared_context.get("start_date") or shared_context.get("end_date")):
        context_hint = (
            f"\nThe timeline should be consistent with the project dates already used "
            f"elsewhere in the document: start {shared_context.get('start_date', 'N/A')}, "
            f"end {shared_context.get('end_date', 'N/A')}."
        )

    prompt = f"""Based on this request: "{user_request}"

Today's date is {today}. The timeline must start on or after today -- never in the past.
{context_hint}
Generate a plausible project timeline. Return ONLY valid JSON, no other text,
in this exact format:
[
  {{"milestone": "Requirements gathering", "start_week": 1, "end_week": 2}},
  {{"milestone": "Design phase", "start_week": 2, "end_week": 4}}
]

Use 4-6 realistic milestones in chronological order, spanning a reasonable
project length (8-16 weeks total). Weeks are relative to the project start date, not today."""
    return _ask_for_json(prompt)


def generate_budget_chart(breakdown: dict, output_path: str):
    """Renders a budget breakdown as a pie chart and saves it as a PNG."""
    labels = list(breakdown.keys())
    values = list(breakdown.values())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Budget Breakdown")
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_timeline_chart(milestones: list, output_path: str):
    """Renders milestones as a horizontal Gantt-style bar chart and saves it as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names = [m["milestone"] for m in milestones]
    starts = [m["start_week"] for m in milestones]
    durations = [m["end_week"] - m["start_week"] for m in milestones]

    y_positions = range(len(names))
    ax.barh(y_positions, durations, left=starts, height=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Week")
    ax.set_title("Project Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def build_charts(user_request: str, document_type: str, output_dir: str, shared_context: dict = None) -> list:
    """
    Generates both charts for a given request and returns a list of
    saved PNG file paths, ready to embed into the docx.
    Any failure in a single chart is skipped rather than crashing the whole run.
    output_dir should be a per-request directory (e.g. a temp dir) so
    concurrent requests never collide or overwrite each other's charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    try:
        budget = get_budget_breakdown(user_request, shared_context)
        budget_path = os.path.join(output_dir, "budget_chart.png")
        generate_budget_chart(budget, budget_path)
        chart_paths.append(budget_path)
    except Exception as e:
        print(f"Skipping budget chart: {e}")

    try:
        milestones = get_timeline_milestones(user_request, shared_context)
        timeline_path = os.path.join(output_dir, "timeline_chart.png")
        generate_timeline_chart(milestones, timeline_path)
        chart_paths.append(timeline_path)
    except Exception as e:
        print(f"Skipping timeline chart: {e}")

    return chart_paths
