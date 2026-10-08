"""Deterministic checks for common requirement defects.

The checks follow the advice in the INCOSE Guide to Writing Requirements and ISO/IEC/IEEE
29148: a requirement should be one verifiable statement, in the imperative "shall", with an
explicit subject, no vague or open-ended wording and nothing left to be decided. Each check
finds a pattern and says why it matters; none of them understands the domain, which is what the
LLM reviewer adds.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

VAGUE = (
    "adequate", "appropriate", "as needed", "as required", "acceptable", "approximately",
    "easy", "easily", "efficient", "efficiently", "fast", "flexible", "intuitive", "minimal",
    "normal", "quick", "quickly", "reasonable", "robust", "seamless", "significant", "simple",
    "sufficient", "suitable", "timely", "user-friendly", "user friendly", "various", "some",
    "several", "many", "most", "high performance", "state of the art", "state-of-the-art",
    "best", "optimal", "effective", "good", "better",
)  # fmt: skip
ESCAPE = (
    "if possible", "as far as possible", "where possible", "where applicable", "if applicable",
    "as appropriate", "if necessary", "if needed", "to the extent possible", "as practical",
    "where appropriate",
)  # fmt: skip
OPEN_ENDED = ("etc", "and so on", "and so forth", "including but not limited to", "such as")
WEAK_MODALS = ("should", "may", "might", "could", "would", "can", "will")
PLACEHOLDERS = ("tbd", "tbc", "tbs", "tba", "to be determined", "to be decided", "xxx")
# Personal pronouns only: "this", "that" and "these" are mostly determiners or relative
# pronouns in requirements ("this function", "data that ...").
PRONOUNS = ("it", "its", "they", "them", "their", "theirs")
SLASH_OK = ("i/o", "n/a", "tcp/ip", "a/c", "client/server")
UNITS = frozenset(
    "s ms us min h d mm cm m km l ml g kg t mg bar pa kpa mpa hz khz mhz v mv a ma w kw mw "
    "b kb mb gb tb bit kbit mbit gbit rpm".split()
)
COMPARATIVES = (
    "faster", "slower", "better", "worse", "more", "less", "fewer", "larger", "smaller",
    "higher", "lower", "easier", "improved", "increased", "reduced",
)  # fmt: skip
IRREGULAR_PARTICIPLES = (
    "built", "done", "given", "kept", "known", "made", "sent", "set", "shown", "taken",
    "written", "held", "found", "run", "put", "read", "chosen", "seen", "brought", "sold",
)  # fmt: skip


@dataclass(frozen=True)
class Finding:
    rule: str
    start: int
    end: int
    text: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _words(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)


_VAGUE = _words(VAGUE)
_ESCAPE = _words(ESCAPE)
_OPEN = re.compile(r"\betc\b\.?|\band so (?:on|forth)\b|\bincluding but not limited to\b",
                   re.IGNORECASE)  # fmt: skip
_WEAK = _words(WEAK_MODALS)
_SHALL = re.compile(r"\bshall\b|\bmust\b", re.IGNORECASE)
_AND_OR = re.compile(r"\b[A-Za-z]+/[A-Za-z]+\b")
_PLACEHOLDER = _words(PLACEHOLDERS)
_PRONOUN = _words(PRONOUNS)
_COMPARATIVE = _words(COMPARATIVES)
_PASSIVE = re.compile(
    r"\b(?:shall|must|will|should|can|may)\s+(?:not\s+)?be\s+(\w+ed|"
    + "|".join(IRREGULAR_PARTICIPLES)
    + r")\b",
    re.IGNORECASE,
)
_BY_AGENT = re.compile(r"\bby\s+(?:the|a|an|each|every|any)?\s*\w+", re.IGNORECASE)
_THAN = re.compile(r"\bthan\b", re.IGNORECASE)


def _hits(pattern: re.Pattern[str], text: str, rule: str, message: str) -> list[Finding]:
    return [Finding(rule, m.start(), m.end(), m.group(0), message) for m in pattern.finditer(text)]


def vague_terms(text: str) -> list[Finding]:
    return _hits(_VAGUE, text, "vague-term", "Not measurable; state a value or criterion.")


def escape_clauses(text: str) -> list[Finding]:
    return _hits(
        _ESCAPE, text, "escape-clause", "Lets the requirement be skipped; state when it applies."
    )


def open_ended(text: str) -> list[Finding]:
    return _hits(_OPEN, text, "open-ended", "Open-ended list; enumerate every item.")


def weak_modal(text: str) -> list[Finding]:
    if _SHALL.search(text):
        return []
    found = _hits(_WEAK, text, "weak-modal", "No binding 'shall'; the statement reads as optional.")
    return found[:1]


def missing_shall(text: str) -> list[Finding]:
    if _SHALL.search(text) or _WEAK.search(text):
        return []
    return [Finding("no-modal", 0, 0, "", "No 'shall': state who shall do what.")]


def combinators(text: str) -> list[Finding]:
    hits = _hits(_AND_OR, text, "and-or", "'and/or' or a slash leaves the scope open.")
    return [
        f
        for f in hits
        if f.text.lower() not in SLASH_OK
        and not all(part.lower() in UNITS for part in f.text.split("/"))
    ]


def placeholders(text: str) -> list[Finding]:
    return _hits(
        _PLACEHOLDER, text, "placeholder", "Undecided content; the requirement is incomplete."
    )


def pronouns(text: str) -> list[Finding]:
    return _hits(_PRONOUN, text, "pronoun", "Pronoun; name what it refers to.")


def passive_without_agent(text: str) -> list[Finding]:
    out = []
    for m in _PASSIVE.finditer(text):
        if not _BY_AGENT.search(text, m.end()):
            message = "Passive voice without an actor; say who or what does it."
            out.append(Finding("passive-no-agent", m.start(), m.end(), m.group(0), message))
    return out


def multiple_requirements(text: str) -> list[Finding]:
    shalls = list(_SHALL.finditer(text))
    if len(shalls) > 1:
        m = shalls[1]
        return [Finding("multiple", m.start(), m.end(), m.group(0),
                        "More than one 'shall'; split into separate requirements.")]  # fmt: skip
    return []


def unbounded_comparatives(text: str) -> list[Finding]:
    if _THAN.search(text):
        return []
    return _hits(
        _COMPARATIVE, text, "comparative", "Comparative without a reference; better than what?"
    )


CHECKS: dict[str, Callable[[str], list[Finding]]] = {
    "vague-term": vague_terms,
    "escape-clause": escape_clauses,
    "open-ended": open_ended,
    "weak-modal": weak_modal,
    "no-modal": missing_shall,
    "and-or": combinators,
    "placeholder": placeholders,
    "pronoun": pronouns,
    "passive-no-agent": passive_without_agent,
    "multiple": multiple_requirements,
    "comparative": unbounded_comparatives,
}


def check(text: str, rules: tuple[str, ...] | None = None) -> list[Finding]:
    out = []
    for name, fn in CHECKS.items():
        if rules is None or name in rules:
            out.extend(fn(text))
    return sorted(out, key=lambda f: (f.start, f.rule))
