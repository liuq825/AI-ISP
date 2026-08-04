import re
from pathlib import Path
from urllib.parse import unquote


def test_relative_markdown_links_resolve() -> None:
    failures: list[str] = []
    for markdown in Path(".").glob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\((\./[^)]+)\)", text):
            decoded = unquote(target.split("#", 1)[0])
            if not (markdown.parent / decoded).exists():
                failures.append(f"{markdown}: {target}")
    for markdown in Path("docs").glob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\((\./[^)]+)\)", text):
            decoded = unquote(target.split("#", 1)[0])
            if not (markdown.parent / decoded).exists():
                failures.append(f"{markdown}: {target}")
    assert failures == []

