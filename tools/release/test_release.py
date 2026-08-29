# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from contextlib import nullcontext
from typing import cast

import pytest
from omegaconf import OmegaConf

from tools.release import release
from tools.release.release import (
    DevReleasePackageInfo,
    LocalPackageInfo,
    Package,
    PackageInfo,
    fail_if_any_target_version_published,
    filter_packages,
    format_dev_release_package_table,
    is_publishable,
    parse_version,
    validate_dev_version,
    validate_local_version,
)


def test_publishable_when_local_version_is_older_than_latest_but_unpublished() -> None:
    info = PackageInfo(
        name="hydra-core",
        local_version=parse_version("1.3.5"),
        latest_version=parse_version("1.4.0"),
        local_version_published=False,
    )

    assert is_publishable(info)


def test_not_publishable_when_local_version_is_already_published() -> None:
    info = PackageInfo(
        name="hydra-core",
        local_version=parse_version("1.3.5"),
        latest_version=parse_version("1.4.0"),
        local_version_published=True,
    )

    assert not is_publishable(info)


def test_validate_local_version_accepts_expected_version() -> None:
    info = LocalPackageInfo(name="hydra-core", local_version=parse_version("1.4.0"))

    validate_local_version(info, parse_version("1.4.0"))


def test_validate_local_version_rejects_unexpected_version() -> None:
    info = LocalPackageInfo(name="hydra-core", local_version=parse_version("1.4.0"))

    with pytest.raises(
        ValueError,
        match="hydra-core version is 1.4.0; expected 1.4.1",
    ):
        validate_local_version(info, parse_version("1.4.1"))


def test_validate_dev_version_accepts_dev_version() -> None:
    validate_dev_version(parse_version("1.4.0.dev3"))


def test_validate_dev_version_rejects_non_dev_version() -> None:
    with pytest.raises(
        ValueError,
        match="Dev releases require a \\.devN version; got 1.4.0",
    ):
        validate_dev_version(parse_version("1.4.0"))


def test_format_dev_release_package_table_includes_publish_plan() -> None:
    table = format_dev_release_package_table(
        [
            DevReleasePackageInfo(
                name="hydra-core",
                local_version=parse_version("1.4.0.dev2"),
                target_version=parse_version("1.4.0.dev3"),
                latest_version=parse_version("1.4.0.dev2"),
                target_version_published=False,
            )
        ]
    )

    assert "Package" in table
    assert "Current local version" in table
    assert "Target version" in table
    assert "PyPI status" in table
    assert "hydra-core" in table
    assert "1.4.0.dev2" in table
    assert "1.4.0.dev3" in table
    assert "not published" in table


def test_fail_if_any_target_version_published_rejects_published_package() -> None:
    infos = [
        DevReleasePackageInfo(
            name="hydra-core",
            local_version=parse_version("1.4.0.dev2"),
            target_version=parse_version("1.4.0.dev3"),
            latest_version=parse_version("1.4.0.dev3"),
            target_version_published=True,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Target version is already published for selected packages: hydra-core",
    ):
        fail_if_any_target_version_published(infos)


def test_filter_packages_keeps_requested_packages_in_order() -> None:
    packages = {
        "hydra": Package(path="."),
        "hydra_optuna_sweeper": Package(path="plugins/hydra_optuna_sweeper"),
        "hydra_ray_launcher": Package(path="plugins/hydra_ray_launcher"),
    }

    selected = filter_packages(packages, " hydra_ray_launcher, hydra_optuna_sweeper ")

    assert list(selected) == ["hydra_ray_launcher", "hydra_optuna_sweeper"]


def test_filter_packages_rejects_unknown_packages() -> None:
    packages = {
        "hydra": Package(path="."),
        "hydra_optuna_sweeper": Package(path="plugins/hydra_optuna_sweeper"),
    }

    with pytest.raises(
        ValueError,
        match=(
            "Unknown package filter: hydra_rq_launcher. "
            "Available packages in selected set: hydra, hydra_optuna_sweeper"
        ),
    ):
        filter_packages(packages, "hydra_rq_launcher")


def test_dispatch_publish_workflow_uses_json_boolean_input(monkeypatch) -> None:
    calls = []

    def fake_run_checked(cmd, cwd=None, stdin=None):
        calls.append((cmd, cwd, stdin))
        return ""

    monkeypatch.setattr(release, "_run_checked", fake_run_checked)
    monkeypatch.setattr(
        release,
        "get_remote_url",
        lambda hydra_root, vcs: "https://github.com/hydra-ecosystem/hydra",
    )

    release.dispatch_publish_workflow(
        "/repo",
        "sl",
        "hydra-full-release",
        parse_version("1.4.0.dev3"),
        "main",
        "a" * 40,
        "hydra_optuna_sweeper",
    )

    assert calls == [
        (
            [
                "gh",
                "workflow",
                "run",
                "publish.yml",
                "--repo",
                "hydra-ecosystem/hydra",
                "--ref",
                "main",
                "--json",
            ],
            "/repo",
            '{"package_set": "hydra-full-release", '
            '"expected_version": "1.4.0.dev3", '
            f'"commit": "{"a" * 40}", '
            '"publish": "true", '
            '"only": "hydra_optuna_sweeper"}',
        )
    ]


def _run_publish_plan(commit: str, head_sha: str = "b" * 40, ref: str = "main"):
    """Drive publish.yml's release-plan script directly.

    The script decides which commit the whole publish run builds, so it is
    exercised here rather than only in CI.
    """
    workflow = pathlib.Path(".github/workflows/publish.yml").read_text()
    match = re.search(r"python <<'PY'\n(.*?)\n\s*PY\n", workflow, re.S)
    assert match is not None, "release-plan script not found in publish.yml"
    script = textwrap.dedent(match.group(1))

    with tempfile.TemporaryDirectory() as tmp:
        output = pathlib.Path(tmp, "output")
        output.touch()
        pathlib.Path(tmp, "summary").touch()
        env = {
            **os.environ,
            "REF_NAME": ref,
            "HEAD_SHA": head_sha,
            "INPUT_COMMIT": commit,
            "INPUT_PACKAGE_SET": "hydra-full-release",
            "INPUT_ONLY": "",
            "INPUT_EXPECTED_VERSION": "1.4.0.dev9",
            "INPUT_PUBLISH": "true",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(pathlib.Path(tmp, "summary")),
        }
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        outputs = dict(
            line.split("=", 1)
            for line in output.read_text().splitlines()
            if "=" in line
        )
    return proc.returncode, outputs.get("checkout_ref")


def test_publish_plan_pins_dispatched_branch_tip_without_a_commit_input() -> None:
    returncode, checkout_ref = _run_publish_plan("")

    assert returncode == 0
    assert checkout_ref == "b" * 40


def test_publish_plan_pins_the_requested_commit() -> None:
    returncode, checkout_ref = _run_publish_plan("f" * 40)

    assert returncode == 0
    assert checkout_ref == "f" * 40


@pytest.mark.parametrize(
    "commit",
    [
        "f67e393af44e",
        "F" * 40,
        "1.3_branch",
        "main; rm -rf /",
    ],
)
def test_publish_plan_rejects_a_commit_that_is_not_a_full_sha(commit: str) -> None:
    returncode, checkout_ref = _run_publish_plan(commit)

    assert returncode == 1
    assert checkout_ref is None


def test_dev_release_dispatches_the_commit_it_reports_after_bumping(
    monkeypatch, caplog
) -> None:
    """The recovery path creates a commit between reporting and dispatching.

    Whatever commit the operator is told about must be the commit that is
    published, so the two must be resolved from the same point in the sequence.
    """
    commits = iter(["a" * 40, "b" * 40])
    current = {"sha": next(commits)}
    dispatched = {}

    monkeypatch.setattr(release, "detect_vcs", lambda hydra_root: "sl")
    monkeypatch.setattr(release, "ensure_publish_tools", lambda *a: None)
    monkeypatch.setattr(release, "ensure_clean_worktree", lambda *a: None)
    monkeypatch.setattr(release, "ensure_publish_base_matches_ref", lambda *a: None)
    monkeypatch.setattr(release, "collect_dev_release_package_info", lambda *a: [])
    monkeypatch.setattr(release, "format_dev_release_package_table", lambda infos: "")
    monkeypatch.setattr(release, "fail_if_any_target_version_published", lambda i: None)
    monkeypatch.setattr(release, "copy_release_workspace", lambda root, tmp: tmp)
    monkeypatch.setattr(release, "validate_dev_release_artifacts", lambda *a: None)
    monkeypatch.setattr(release, "set_package_versions", lambda *a: None)
    monkeypatch.setattr(
        release, "get_worktree_status", lambda *a: "M hydra/__init__.py"
    )
    monkeypatch.setattr(release, "push_current_ref", lambda *a: None)
    monkeypatch.setattr(
        release, "get_current_commit", lambda hydra_root, vcs: current["sha"]
    )

    def fake_commit_dev_release(hydra_root, vcs, target_version):
        current["sha"] = next(commits)

    def fake_dispatch(hydra_root, vcs, package_set, target_version, ref, commit, only):
        dispatched["commit"] = commit

    monkeypatch.setattr(release, "commit_dev_release", fake_commit_dev_release)
    monkeypatch.setattr(release, "dispatch_publish_workflow", fake_dispatch)

    cfg = cast(
        release.Config,
        OmegaConf.create(
            {
                "version": "1.4.0.dev3",
                "dry_run": False,
                "publish": True,
                "workflow_ref": "main",
                "only": "",
                "packages": {},
                "repository": {"name": "pypi"},
            }
        ),
    )

    with caplog.at_level(logging.INFO, logger=release.log.name):
        release.run_dev_release(cfg, "/repo", pathlib.Path("/repo/build"), "hydra-core")

    reported = [
        line.split("commit=")[1].split()[0]
        for line in caplog.messages
        if "commit=" in line
    ]

    assert dispatched["commit"] == "b" * 40
    assert reported == ["b" * 40]


def test_check_build_artifacts_upgrades_smoke_environment_pip(
    monkeypatch, tmp_path
) -> None:
    wheel = tmp_path / "package.whl"
    wheel.touch()
    smoke_dir = tmp_path / "smoke"
    calls = []

    def fake_run_checked(cmd, cwd=None, stdin=None):
        calls.append(cmd)
        return ""

    monkeypatch.setattr(release, "_run_checked", fake_run_checked)
    monkeypatch.setattr(
        release.tempfile,
        "TemporaryDirectory",
        lambda prefix: nullcontext(str(smoke_dir)),
    )

    release.check_build_artifacts(tmp_path)

    venv_path = smoke_dir / "venv"
    smoke_python = release._python_bin(venv_path)
    assert calls == [
        [release.sys.executable, "-m", "twine", "check", str(wheel)],
        [release.sys.executable, "-m", "venv", str(venv_path)],
        [str(smoke_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(smoke_python), "-m", "pip", "install", "--no-deps", str(wheel)],
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "https://github.com/hydra-ecosystem/hydra",
            "https://github.com/hydra-ecosystem/hydra",
        ),
        (
            "default = https://github.com/hydra-ecosystem/hydra",
            "https://github.com/hydra-ecosystem/hydra",
        ),
    ],
)
def test_get_remote_url_accepts_sapling_path_output_formats(
    monkeypatch, output, expected
) -> None:
    monkeypatch.setattr(release, "_single_line", lambda cmd, cwd: output)

    assert release.get_remote_url("/repo", "sl") == expected


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://github.com/hydra-ecosystem/hydra",
        "https://github.com/hydra-ecosystem/hydra.git",
        "ssh://git@github.com/hydra-ecosystem/hydra",
        "ssh://git@github.com/hydra-ecosystem/hydra.git",
        "git@github.com:hydra-ecosystem/hydra.git",
    ],
)
def test_get_github_repo_slug_accepts_common_github_remote_urls(remote_url) -> None:
    assert release.get_github_repo_slug(remote_url) == "hydra-ecosystem/hydra"
