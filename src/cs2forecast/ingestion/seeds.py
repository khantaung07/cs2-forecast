from pathlib import Path


def read_seed_titles(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Seed file does not exist: {path}")

    titles: list[str] = []
    seen: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        title = line.strip()

        if not title:
            continue

        if title.startswith("#"):
            continue

        if title in seen:
            continue

        titles.append(title)
        seen.add(title)

    return titles