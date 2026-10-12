"""Check tracked paths and built distributions without reading private file contents.

Run with uv run scripts/check_release.py [--artifacts DIRECTORY]. This is a
path/metadata gate, not a credential scanner: review changes to public files too.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tomllib
import zipfile


PUBLIC_KNOWLEDGE = {"knowledge/README.md", "knowledge/getting-started.md"}
PUBLIC_CONFIG = {
    "config/arietta_voice.toml", "config/arietta_voice.toml.dist",
    "config/arietta_voice.toml.example", "config/SOUL.md", "config/MEMORY.md.example",
}
PRIVATE_DIRECTORIES = {"logs", ".knowledge_index", ".private-backups", ".venv", "tmp", ".aws"}


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return True
    if path.parts and path.parts[0] == "knowledge" and name not in PUBLIC_KNOWLEDGE:
        return True
    if path.parts and path.parts[0] == "config" and name not in PUBLIC_CONFIG:
        return True
    return (
        bool(PRIVATE_DIRECTORIES.intersection(path.parts))
        or path.name == "MEMORY.md"
        or path.name in {".env", "credentials", ".DS_Store"}
        or path.name.startswith(".env.") and path.name != ".env.example"
        or name.endswith((".local.toml", ".local.md", ".admin.json", ".working", ".bak"))
    )


def check_tracked(root: Path) -> list[str]:
    records = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    errors = []
    for record in filter(None, records):
        metadata, name = record.split("\t", 1)
        if private_path(name):
            errors.append(f"Private path is tracked: {name}")
        elif metadata.startswith("120000 ") and name in PUBLIC_CONFIG | PUBLIC_KNOWLEDGE:
            errors.append(f"Public starter file must not be a symlink: {name}")
    return errors


def check_versions(root: Path) -> tuple[str, list[str]]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    versions = [p["version"] for p in lock["package"] if p["name"] == project["name"]]
    return project["version"], ([] if versions == [project["version"]] else [
        "Project and uv.lock versions differ; run uv lock and uv sync --locked."
    ])


def _metadata_errors(data: bytes, expected_version: str) -> list[str]:
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"] != "arietta-voice" or metadata["Version"] != expected_version:
        return ["Distribution name/version does not match pyproject.toml."]
    return []


def check_wheel(path: Path, expected_version: str) -> list[str]:
    errors = []
    metadata = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = member.filename
            parts = PurePosixPath(name).parts
            allowed = (
                len(parts) > 1 and parts[0] == "arietta_voice" and name.endswith(".py")
                or len(parts) > 1 and parts[0] == f"arietta_voice-{expected_version}.dist-info"
            )
            if private_path(name) or not allowed or (member.external_attr >> 16) & 0o170000 == 0o120000:
                errors.append(f"Unexpected wheel member: {name}")
            if name.endswith(".dist-info/METADATA"):
                metadata.append(archive.read(member))
        if len(metadata) != 1:
            errors.append("Wheel must contain exactly one METADATA file.")
        else:
            errors.extend(_metadata_errors(metadata[0], expected_version))
    return errors


def check_sdist(path: Path, expected_version: str) -> list[str]:
    errors = []
    metadata = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != f"arietta_voice-{expected_version}":
                errors.append(f"Unexpected sdist root: {member.name}")
                continue
            name = "/".join(parts[1:])
            if private_path(name) or not (member.isdir() or member.isfile()):
                errors.append(f"Unexpected sdist member: {member.name}")
            if name == "PKG-INFO" and member.isfile():
                metadata.append(archive.extractfile(member).read())
        if len(metadata) != 1:
            errors.append("Sdist must contain exactly one root PKG-INFO file.")
        else:
            errors.extend(_metadata_errors(metadata[0], expected_version))
    return errors


def check_artifacts(directory: Path, expected_version: str) -> list[str]:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return ["Artifact directory must contain exactly one wheel and one sdist."]
    try:
        return check_wheel(wheels[0], expected_version) + check_sdist(sdists[0], expected_version)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return [f"Cannot inspect distributions ({type(exc).__name__})."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version, errors = check_versions(root)
    errors.extend(check_tracked(root))
    if arguments.artifacts:
        errors.extend(check_artifacts(arguments.artifacts, version))
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Release checks passed for {version} (tracked paths, lockfile"
          + (", wheel and sdist)." if arguments.artifacts else ")."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
