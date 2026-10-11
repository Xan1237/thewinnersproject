"""Episode persistence as JSON Lines: one episode per line."""

from __future__ import annotations

from pathlib import Path

from winners.schema import Episode


def save_episodes(episodes: list[Episode], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(ep.model_dump_json() + "\n")
    return path


def load_episodes(path: str | Path) -> list[Episode]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [Episode.model_validate_json(line) for line in f if line.strip()]
