"""Static checks of the console's frontend, which has no build step to catch mistakes.

Every imported name must be exported by its module, every element the scripts
look up by a fixed id must exist in the page, and each module must parse as
JavaScript (when Node is installed, as it is on the CI runners).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "arietta_voice" / "web" / "static"
SCRIPTS = sorted((STATIC / "js").glob("*.js"))
IMPORT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./([\w-]+\.js)"', re.DOTALL)
EXPORT = re.compile(r"export\s+(?:async\s+)?(?:function\*?|const|let|class)\s+([A-Za-z_$][\w$]*)")
FIXED_ID = re.compile(r'\$\("#([A-Za-z][\w-]*)"\)|getElementById\("([A-Za-z][\w-]*)"\)')


def exports(path: Path) -> set[str]:
    return set(EXPORT.findall(path.read_text(encoding="utf-8")))


def test_there_are_console_scripts():
    assert len(SCRIPTS) >= 8
    assert (STATIC / "js" / "main.js") in SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_every_imported_name_is_exported(script):
    for names, module in IMPORT.findall(script.read_text(encoding="utf-8")):
        target = STATIC / "js" / module
        assert target.exists(), f"{script.name} imports missing module {module}"
        wanted = {name.split(" as ")[0].strip() for name in names.split(",") if name.strip()}
        missing = wanted - exports(target)
        assert not missing, f"{script.name} imports {sorted(missing)} that {module} does not export"


def test_every_element_looked_up_by_id_is_in_the_page():
    page_ids = set(re.findall(r'\bid="([^"]+)"', (STATIC / "index.html").read_text("utf-8")))
    missing = {}
    for script in SCRIPTS:
        for pair in FIXED_ID.findall(script.read_text(encoding="utf-8")):
            element = next(part for part in pair if part)
            if element not in page_ids:
                missing.setdefault(script.name, []).append(element)
    assert not missing, f"Elements looked up but not in index.html: {missing}"


def test_the_page_loads_only_existing_scripts_and_styles():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    for reference in re.findall(r'(?:src|href)="/static/([^"]+)"', page):
        assert (STATIC / reference).exists(), f"index.html refers to missing {reference}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_each_module_parses(script, tmp_path):
    # --check parses without running; the .mjs name makes Node read it as a module.
    module = tmp_path / f"{script.stem}.mjs"
    module.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(module)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
