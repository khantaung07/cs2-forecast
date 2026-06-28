import hashlib
import re


def clean_text(value: object) -> str:
    text = str(value).strip()

    # Remove simple wiki links: [[G2 Esports|G2]] -> G2, [[Vitality]] -> Vitality
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Remove basic formatting
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


def slugify(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    readable = "_".join(slugify(part) for part in parts if part)[:80]
    return f"{readable}_{digest}"


def team_id_from_name(name: str) -> str:
    return slugify(name)

TEAM_ID_ALIASES = {
    # Vitality
    "vitality": "vit",
    "team_vitality": "vit",

    # Liquid
    "liquid": "tl",
    "team_liquid": "tl",

    # Complexity
    "complexity": "col",
    "complexity_gaming": "col",

    # Eternal Fire
    "eternal_fire": "ef",

    # The MongolZ
    "the_mongolz": "mongolz",

    # GamerLegion
    "gamerlegion": "gl",
    "gamer_legion": "gl",

    # Virtus.pro
    "virtus_pro": "vp",
    "virtuspro": "vp",

    # NAVI
    "natus_vincere": "navi",
    "navi": "navi",

    # FaZe
    "faze_clan": "faze",

    # 3DMAX
    "3d_max": "3dmax",

    # paiN
    "pain_gaming": "pain",
}


def canonical_team_id(value: str) -> str:
    raw_id = team_id_from_name(value)
    return TEAM_ID_ALIASES.get(raw_id, raw_id)