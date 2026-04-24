"""
Long Chain-of-Thought (CoT) Data Augmentation for SFT Training.

Inspired by WebAgent-R1: enriches oracle (observation, action) pairs with
detailed reasoning traces. The SFT model then learns to THINK before acting,
which WebAgent-R1 showed is the single biggest performance driver.

Approach: Template-based reasoning (no LLM API needed — runs locally, free).
For each oracle step, generates a first-person reasoning trace that:
  1. Describes what the agent observes on the page
  2. Identifies the current task state (what's done, what's next)
  3. Explains WHY this specific action is the right next step
  4. Names the specific element being interacted with

Usage:
    python training/augment_cot.py
    python training/augment_cot.py --input training/data/oracle_trajectories.jsonl --output training/data/oracle_cot.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _extract_elements(a11y_tree: str) -> dict[str, dict]:
    """Parse a11y tree text to extract element info keyed by ref."""
    elements = {}
    for line in a11y_tree.split("\n"):
        ref_match = re.search(r"ref=(\w+)", line)
        if not ref_match:
            continue
        ref = ref_match.group(1)
        role_match = re.search(r"role=(\w+)", line)
        name_match = re.search(r'name="([^"]*)"', line)
        value_match = re.search(r'value="([^"]*)"', line)
        elements[ref] = {
            "role": role_match.group(1) if role_match else "",
            "name": name_match.group(1) if name_match else "",
            "value": value_match.group(1) if value_match else "",
        }
    return elements


def _describe_page_state(elements: dict[str, dict]) -> str:
    """Generate a brief description of the visible page state."""
    form_fields = []
    buttons = []
    for ref, info in elements.items():
        role = info["role"]
        name = info["name"]
        value = info["value"]
        if role in ("textbox", "searchbox"):
            status = f'"{value}"' if value else "empty"
            form_fields.append(f'{name} ({ref}): {status}')
        elif role == "combobox":
            form_fields.append(f'{name} ({ref}): selected="{value}"')
        elif role == "button":
            buttons.append(f'{name} ({ref})')
    parts = []
    if form_fields:
        parts.append("Form fields: " + "; ".join(form_fields[:6]))
    if buttons:
        parts.append("Buttons: " + ", ".join(buttons[:5]))
    return ". ".join(parts)


def _generate_reasoning(record: dict, elements: dict[str, dict]) -> str:
    """Generate a first-person reasoning trace for an oracle action."""
    action = record["action"]
    atype = action.get("action_type", "wait")
    ref = action.get("element_ref", "")
    text_val = action.get("text_value", "")
    instruction = record.get("instruction", "")
    step = record.get("step", 0)

    el_info = elements.get(ref, {})
    el_name = el_info.get("name", ref)
    el_role = el_info.get("role", "element")
    el_value = el_info.get("value", "")

    page_state = _describe_page_state(elements)

    # Build context about what's already filled
    filled = []
    empty = []
    for r, info in elements.items():
        if info["role"] in ("textbox", "searchbox"):
            if info["value"]:
                filled.append(f'{info["name"]}="{info["value"]}"')
            else:
                empty.append(info["name"])
        elif info["role"] == "combobox":
            if info["value"] and "select" not in info["value"].lower():
                filled.append(f'{info["name"]}="{info["value"]}"')
            else:
                empty.append(info["name"])

    # Detect distractor buttons (cookie banners, promos)
    distractor_names = ("accept", "cookie", "close", "dismiss", "promo", "claim")

    if atype == "click":
        name_lower = el_name.lower()
        if any(d in name_lower for d in distractor_names):
            reasoning = (
                f"I notice there's a popup or banner on the page — "
                f'a button labeled "{el_name}" ({ref}). '
                f"I need to dismiss this before I can interact with the main form. "
                f"Clicking {ref} to clear it."
            )
        elif any(kw in name_lower for kw in ("search", "find", "go", "check", "look")):
            filled_str = ", ".join(filled) if filled else "the required fields"
            reasoning = (
                f"I've filled in {filled_str}. "
                f"The form is ready to submit. "
                f'I see the "{el_name}" button ({ref}) which will submit the search. '
                f"Clicking it to proceed to results."
            )
        elif any(kw in name_lower for kw in ("book", "reserve", "purchase", "buy", "secure")):
            reasoning = (
                f"The search results are showing. "
                f'I can see a "{el_name}" button ({ref}). '
                f"This will initiate the booking. Clicking it now."
            )
        elif any(kw in name_lower for kw in ("confirm", "complete", "finalize", "place")):
            reasoning = (
                f"The booking details are displayed. "
                f'I see the "{el_name}" button ({ref}) to finalize. '
                f"Clicking to confirm the booking and complete the task."
            )
        else:
            reasoning = (
                f'I see a button "{el_name}" ({ref}). '
                f"Based on the current task state, clicking this is the next step. "
                f"Proceeding."
            )

    elif atype == "type":
        if el_value and text_val.lower() in el_value.lower():
            reasoning = (
                f'The "{el_name}" field ({ref}) already contains "{el_value}". '
                f"But I need to ensure it has '{text_val}'. "
                f"Typing to set the correct value."
            )
        else:
            # Determine what field this is semantically
            name_lower = el_name.lower()
            if any(kw in name_lower for kw in ("from", "origin", "depart", "start")):
                field_purpose = "origin/departure"
            elif any(kw in name_lower for kw in ("to", "destination", "arrive", "going")):
                field_purpose = "destination/arrival"
            elif any(kw in name_lower for kw in ("class", "cabin", "fare", "seat")):
                field_purpose = "travel class"
            else:
                field_purpose = el_name

            empty_note = ""
            if empty:
                remaining = [f for f in empty if f.lower() != el_name.lower()]
                if remaining:
                    empty_note = f" After this, I still need to fill: {', '.join(remaining[:3])}."

            reasoning = (
                f"Looking at the form, the {field_purpose} field \"{el_name}\" ({ref}) is "
                f'{"empty" if not el_value else f"set to \"{el_value}\""}. '
                f"My task says to book from/to specific cities, so I need to type '{text_val}' here."
                f"{empty_note}"
            )

    elif atype == "scroll":
        delta = action.get("scroll_delta", 0)
        direction = "down" if delta > 0 else "up"
        reasoning = (
            f"I can't see the elements I need on the current viewport. "
            f"Scrolling {direction} to reveal more of the page."
        )

    elif atype == "wait":
        reasoning = (
            "The page may be loading or transitioning. "
            "Waiting briefly before taking the next action."
        )

    else:
        reasoning = f"Taking action: {atype}."

    # Add task context if step 0
    if step == 0:
        reasoning = f"Starting task: {instruction}. " + reasoning

    return reasoning


def augment_file(input_path: str, output_path: str) -> None:
    """Read oracle JSONL, add CoT reasoning, write augmented JSONL."""
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        count = 0
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            a11y_tree = record.get("observation", "")
            elements = _extract_elements(a11y_tree)
            reasoning = _generate_reasoning(record, elements)

            # Add the reasoning as a new field
            record["reasoning"] = reasoning

            # Build the think-then-act assistant response
            action = record["action"].copy()
            for key in list(action.keys()):
                if key not in ("action_type", "element_ref", "text_value", "scroll_delta"):
                    del action[key]
            action.setdefault("scroll_delta", 0)
            action.setdefault("text_value", "")
            action.setdefault("element_ref", "")

            record["cot_response"] = (
                f"<think>\n{reasoning}\n</think>\n"
                f"<answer>\n{json.dumps(action, separators=(',', ':'))}\n</answer>"
            )

            fout.write(json.dumps(record) + "\n")
            count += 1

    print(f"Augmented {count} records: {input_path} -> {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "training/data/oracle_trajectories.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training/data/oracle_cot.jsonl"
    augment_file(input_file, output_file)
