"""Prompt Composer: joins prompt fragments into one well-formed prompt
string — Sprint 4 Prompt 12.

Replaces `prompt_generator.py`'s previous inline joining logic
(`", ".join(clause.strip().rstrip(",") for clause in clauses if
clause.strip())`), which only stripped trailing *commas* from each
fragment before rejoining. Any fragment that was itself a complete
sentence — `shot.description` and `shot.environment` both can be,
since they're built from freeform narrative text
(`narrative_generator.py`'s screenplay output) — could already end in
a period, producing "sentence., next fragment" when the outer join
added its own comma. Confirmed directly, not assumed: reproduced the
exact "arena.," defect from the real three-shot GPU evidence
(Sprint 4 Prompt 12) from first principles by running the actual
pipeline — `shot.description` = "...in temple ruins — a grueling
dueling arena." (trailing period, from timeline_engine.py's own
f-string), immediately followed by the next fragment via ", ".

The fix generalizes rather than patches the one observed string: strip
*any* trailing sentence-ending punctuation from every fragment, not
just commas, before rejoining — correct for any fragment regardless of
whether it happens to be a bare phrase or a complete sentence.
"""

from __future__ import annotations

from collections.abc import Sequence

# Every character a fragment might legitimately end with that would
# collide with the outer ", " join if left in place. Not just "," —
# the bug this module fixes was specifically that only commas were
# stripped, missing periods and semicolons a complete-sentence
# fragment (like shot.description) can end with.
_TRAILING_PUNCTUATION = ".,;: "


def compose_prompt(fragments: Sequence[str]) -> str:
    """Joins prompt fragments into one deterministic, well-formed
    prompt string.

    Guarantees:
    - proper spacing (each fragment trimmed before joining);
    - no dangling/duplicate punctuation (any trailing '.', ',', ';',
      ':' is stripped from each fragment first — not just commas);
    - no accidental empty fragments (blank/whitespace-only fragments
      are dropped, never producing ", ," from an empty entry);
    - deterministic output (pure function of `fragments`, no
      randomness, no external state);
    - stable fragment ordering (exactly the order given — this
      function never reorders, sorts, or deduplicates fragments by
      content).

    Does not add an LLM call or any semantic rewriting — this is
    string hygiene only, exactly the fragments given, cleaned and
    joined.
    """
    cleaned: list[str] = []
    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped:
            continue
        stripped = stripped.rstrip(_TRAILING_PUNCTUATION).strip()
        if stripped:
            cleaned.append(stripped)
    return ", ".join(cleaned)


def compose_prompt_from_blocks(*blocks: Sequence[str]) -> str:
    """Same guarantees as `compose_prompt`, for the common case of
    composing from several named fragment groups (e.g. a stable
    continuity block, a shot-action block, a shot-camera block, and a
    global style block — see `prompt_generator.py`'s own explicit
    four-block structure) rather than one flat fragment list. Purely a
    convenience — flattens `blocks` in the order given and delegates
    to `compose_prompt`; block boundaries carry no special joining
    behavior of their own.
    """
    flattened: list[str] = []
    for block in blocks:
        flattened.extend(block)
    return compose_prompt(flattened)
