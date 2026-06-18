"""Tests for named Mini-Me profiles."""

from __future__ import annotations

from chess_mini_me import training


def test_no_profiles_initially(tmp_path, monkeypatch) -> None:
    """A fresh data directory should report no profiles."""
    monkeypatch.setenv("CHESS_MINI_ME_DATA", str(tmp_path))
    assert training.list_profile_names() == []


def test_profile_names_are_made_filesystem_safe() -> None:
    """Profile names should be cleaned of awkward characters."""
    assert training.safe_profile_name("Magnus Carlsen!") == "Magnus-Carlsen"
    assert training.safe_profile_name("   ") == "profile"
    assert training.safe_profile_name("a/b\\c") == "a-b-c"


def test_saved_profiles_are_listed(tmp_path, monkeypatch) -> None:
    """A profile is listed once it has a dataset or a model on disk."""
    monkeypatch.setenv("CHESS_MINI_ME_DATA", str(tmp_path))
    store = training.store_for_profile("my-style")
    # An empty directory should not yet count as a profile.
    assert training.list_profile_names() == []

    store.dataset_path.write_bytes(b"placeholder")
    assert training.list_profile_names() == ["my-style"]

    training.store_for_profile("Magnus").dataset_path.write_bytes(b"placeholder")
    assert training.list_profile_names() == ["Magnus", "my-style"]


def test_store_for_profile_is_under_the_profiles_root(tmp_path, monkeypatch) -> None:
    """A profile's store should live inside the profiles directory."""
    monkeypatch.setenv("CHESS_MINI_ME_DATA", str(tmp_path))
    store = training.store_for_profile("test")
    assert store.directory.parent == training.profiles_root()
    assert store.directory.parent.parent == tmp_path
