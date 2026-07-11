"""Prompt contract between the router and the Dynamo code model.

PROMPT_TEMPLATE is the only interface the two models share: the router's
(sanitized) instruction is pasted into it, the result is encoded with
Dynamo's BPE tokenizer, and the model continues with code. SFT builds the
identical prefix with the loss masked to the code tokens, so any change to
the template invalidates existing SFT checkpoints — keep this module the
single definition and import it everywhere.
"""
from __future__ import annotations
import re

PROMPT_TEMPLATE = "User request: {user_request}\nRouter output: {precise_instruction}\n"

# "Sure, here's the instruction:", "Instruction:", "Here is a precise
# technical instruction -" and similar chat-model throat-clearing.
_PREAMBLE_RE = re.compile(
    r"^(?:(?:sure|certainly|okay|of course)[,.!]?\s+)?"
    r"(?:here(?:'s| is)(?: the| an?| your)?\s+)?"
    r"(?:single\s+)?(?:precise\s+)?(?:technical\s+)?"
    r"(?:instruction|prompt)\s*[:\-]\s*",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^```[\w+-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def build_prompt(user_request: str, precise_instruction: str) -> str:
    return PROMPT_TEMPLATE.format(
        user_request=user_request,
        precise_instruction=precise_instruction,
    )


def sanitize_router_output(text: str, max_chars: int = 1200) -> str:
    """Normalize the router's raw generation to the single-line instruction
    the template expects.

    The router is a small instruct model and, despite the system prompt,
    sometimes wraps its answer in a code fence or quotes, prepends a
    preamble, or appends an explanation after a blank line. The template is
    line-oriented ("Router output: {x}\\n"), so stray newlines would also
    corrupt the format Dynamo was fine-tuned on. Returns "" if nothing
    usable remains — callers should fall back to the raw user request.
    """
    cleaned = text.strip()

    fenced = _FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    cleaned = _PREAMBLE_RE.sub("", cleaned, count=1).lstrip()

    # a single instruction lives in the first paragraph; later paragraphs
    # are almost always commentary
    cleaned = cleaned.split("\n\n", 1)[0].strip()

    if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
        cleaned = cleaned[1:-1].strip()

    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars].rsplit(" ", 1)[0]
        cleaned = (cut or cleaned[:max_chars]).rstrip()

    return cleaned
