"""Room Matching Agent.

Normalizes the free-text room names found across OTAs and hotel sites
(``"Deluxe King"``, ``"King Deluxe"``, ``"Premium King Roo"``) into the canonical
:class:`~app.schemas.common.RoomType` vocabulary. A fast, deterministic keyword
matcher resolves the overwhelming majority of cases; only genuinely ambiguous
names fall through to the LLM (when enabled), and any remaining unknowns map to
``RoomType.OTHER`` so the pipeline never raises on unexpected input.
"""

from __future__ import annotations

import re

from app.agents.llm import LLMClient
from app.schemas.common import RoomType
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger

logger = get_logger("agent")

# Common scraped misspellings / abbreviations normalized before matching.
_TYPO_FIXES = {
    "roo": "room",
    "rom": "room",
    "ste": "suite",
    "ste.": "suite",
    "std": "standard",
    "std.": "standard",
    "dlx": "deluxe",
    "exec": "executive",
    "jr": "junior",
    "ada": "accessible",
}

_BED_KEYWORDS = {
    "king": "king",
    "queen": "queen",
    "twin": "twin",
    "double double": "twin",
    "two double": "twin",
    "two queen": "twin",
}

_TIER_KEYWORDS = {
    "executive": "executive",
    "junior": "junior",
    "presidential": "executive",
    "deluxe": "deluxe",
    "premium": "deluxe",
    "superior": "deluxe",
    "luxury": "deluxe",
    "family": "family",
    "accessible": "accessible",
    "wheelchair": "accessible",
    "standard": "standard",
    "classic": "standard",
    "economy": "standard",
}

_TIER_BED_MAP: dict[tuple[str, str], RoomType] = {
    ("standard", "queen"): RoomType.STANDARD_QUEEN,
    ("standard", "king"): RoomType.STANDARD_KING,
    ("standard", "twin"): RoomType.STANDARD_TWIN,
    ("deluxe", "queen"): RoomType.DELUXE_QUEEN,
    ("deluxe", "king"): RoomType.DELUXE_KING,
    ("deluxe", "twin"): RoomType.STANDARD_TWIN,
}


class RoomMatcher:
    """Resolve raw room names to canonical :class:`RoomType` values."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm
        self._cache: dict[str, RoomType] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase, de-punctuate, fix typos, and collapse whitespace."""
        text = re.sub(r"[^a-z0-9\s]", " ", name.lower())
        tokens = [_TYPO_FIXES.get(tok, tok) for tok in text.split()]
        return " ".join(tokens).strip()

    def _rule_match(self, normalized: str) -> RoomType:
        """Deterministic keyword match; returns ``OTHER`` when inconclusive."""
        tier = next(
            (t for kw, t in _TIER_KEYWORDS.items() if kw in normalized), None
        )
        bed = next((b for kw, b in _BED_KEYWORDS.items() if kw in normalized), None)

        if tier == "executive":
            return RoomType.EXECUTIVE_SUITE
        if tier == "junior":
            return RoomType.JUNIOR_SUITE
        if tier == "family":
            return RoomType.FAMILY_ROOM
        if tier == "accessible":
            return RoomType.ACCESSIBLE
        if "suite" in normalized and tier == "deluxe":
            return RoomType.JUNIOR_SUITE
        if tier and bed:
            return _TIER_BED_MAP.get((tier, bed), RoomType.OTHER)
        if bed:
            return _TIER_BED_MAP.get(("standard", bed), RoomType.OTHER)
        return RoomType.OTHER

    def _llm_match(self, raw_name: str) -> RoomType:
        """Ask the LLM to classify a name into the enum; tolerate failure."""
        if self._llm is None or not self._llm.enabled:
            return RoomType.OTHER
        options = ", ".join(rt.value for rt in RoomType)
        try:
            payload = self._llm.complete_json(
                system=(
                    "You normalize hotel room names into a fixed vocabulary. "
                    f"Choose exactly one of: {options}."
                ),
                user=(
                    f'Room name: "{raw_name}". '
                    'Return {"room_type": "<ONE_OF_THE_OPTIONS>"}.'
                ),
            )
            value = str(payload.get("room_type", "")).strip().upper()
            return RoomType(value)
        except (LLMError, ValueError) as exc:
            logger.warning("LLM room match failed for {!r}: {}", raw_name, exc)
            return RoomType.OTHER

    def match(self, raw_name: str) -> RoomType:
        """Return the canonical room type for a raw name (cached)."""
        if not raw_name or not raw_name.strip():
            return RoomType.OTHER
        if raw_name in self._cache:
            return self._cache[raw_name]
        normalized = self._normalize(raw_name)
        result = self._rule_match(normalized)
        if result is RoomType.OTHER:
            result = self._llm_match(raw_name)
        self._cache[raw_name] = result
        return result

    def match_many(self, raw_names: list[str]) -> dict[str, RoomType]:
        """Resolve a batch of names, returning a name -> RoomType mapping."""
        return {name: self.match(name) for name in raw_names}
