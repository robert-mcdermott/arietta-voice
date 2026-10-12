import io
import subprocess
import tarfile
import zipfile

import pytest

from scripts.check_release import (
    PUBLIC_CONFIG, PUBLIC_KNOWLEDGE, check_artifacts, check_tracked, check_versions,
)


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def stage(root, names):
    for name in names:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Synthetic fixture only\n")
    subprocess.run(["git", "add", "--force", "--", *names], cwd=root, check=True)


def test_public_starter_files_are_allowed(repository):
    stage(repository, sorted(PUBLIC_CONFIG | PUBLIC_KNOWLEDGE))
    assert check_tracked(repository) == []


@pytest.mark.parametrize("name", [
    "knowledge/custom.md", "knowledge/nested/custom.md", "knowledge/photo.png",
    "config/arietta_voice.local.toml", "config/SOUL.local.md", "config/arietta_voice.admin.json",
    "config/.private-backups/settings.toml", "config/arietta_voice.toml.working",
    "config/MEMORY.md", "logs/events.jsonl", "logs/recording.wav",
    ".knowledge_index/vectors.npy", ".env", ".aws/credentials",
])
def test_force_added_private_files_fail_without_deleting_local_data(repository, name):
    (repository / ".gitignore").write_text("*\n")
    stage(repository, [name])
    assert check_tracked(repository) == [f"Private path is tracked: {name}"]
    assert (repository / name).read_text() == "Synthetic fixture only\n"


def test_untracked_private_files_do_not_fail(repository):
    (repository / "config").mkdir()
    (repository / "config" / "arietta_voice.local.toml").write_text("synthetic")
    assert check_tracked(repository) == []


def test_starter_symlink_cannot_bypass_allowlist(repository):
    (repository / "knowledge").mkdir()
    (repository / "private.md").write_text("synthetic")
    (repository / "knowledge" / "README.md").symlink_to("../private.md")
    subprocess.run(["git", "add", "knowledge/README.md"], cwd=repository, check=True)
    assert "must not be a symlink" in check_tracked(repository)[0]


def test_lockfile_version_must_match_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="arietta-voice"\nversion="0.1.0"\n')
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="arietta-voice"\nversion="0.1.0"\n')
    assert check_versions(tmp_path) == ("0.1.0", [])
    lock.write_text('[[package]]\nname="arietta-voice"\nversion="0.2.0"\n')
    assert check_versions(tmp_path)[1]


def distributions(directory, *, version="0.1.0", wheel_extra=None, sdist_extra=None, symlink=False):
    metadata = f"Name: arietta-voice\nVersion: {version}\n".encode()
    with zipfile.ZipFile(directory / "arietta_voice-0.1.0-py3-none-any.whl", "w") as archive:
        archive.writestr("arietta_voice/__init__.py", "")
        archive.writestr("arietta_voice-0.1.0.dist-info/METADATA", metadata)
        if wheel_extra:
            archive.writestr(wheel_extra, "synthetic")
    with tarfile.open(directory / "arietta_voice-0.1.0.tar.gz", "w:gz") as archive:
        files = {"PKG-INFO": metadata, "src/arietta_voice/__init__.py": b""}
        if sdist_extra:
            files[sdist_extra] = b"synthetic"
        for name, data in files.items():
            member = tarfile.TarInfo(f"arietta_voice-0.1.0/{name}")
            member.size = len(data)
            if name == sdist_extra and symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../private.md"
                member.size = 0
            archive.addfile(member, io.BytesIO(data))


def test_distribution_metadata_and_private_paths(tmp_path):
    distributions(tmp_path)
    assert check_artifacts(tmp_path, "0.1.0") == []


@pytest.mark.parametrize("options", [
    {"version": "0.2.0"},
    {"wheel_extra": "arietta_voice/SOUL.local.md"},
    {"wheel_extra": "arietta_voice/../../private.txt"},
    {"sdist_extra": "config/arietta_voice.local.toml"},
    {"sdist_extra": "knowledge/custom.md"},
    {"sdist_extra": "logs/recording.wav"},
    {"sdist_extra": "src/link.py", "symlink": True},
])
def test_distributions_reject_private_members_and_version_drift(tmp_path, options):
    distributions(tmp_path, **options)
    assert check_artifacts(tmp_path, "0.1.0")


def test_distribution_directory_must_be_complete(tmp_path):
    assert check_artifacts(tmp_path, "0.1.0")
