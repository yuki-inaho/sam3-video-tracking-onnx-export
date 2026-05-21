"""Pytest configuration and shared fixtures.

Session-level isolation helpers ensure that tests importing sam3 via
different loading strategies (equiv-source sys.path injection vs. installed
editable package) do not leak module state into each other.
"""

from __future__ import annotations

import sys
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _reset_sam3_module_cache() -> Generator[None, None, None]:
    """Auto-use fixture: snapshot and restore sam3.* sys.modules entries.

    Motivation (D5-3):
        test_equiv_detector_matches_baseline.py loads the patched model_builder
        via importlib under a ``sam3_equiv_test.*`` key, intentionally leaving
        the official ``sam3.*`` modules intact.
        test_image_encoder_onnx.py calls ``_evict_sam3_cache()`` which deletes
        all ``sam3.*`` entries and re-imports from the equiv source.
        Without isolation, the second test's cleanup can invalidate modules
        that the first test's ``scope="module"`` fixture is still holding
        references to, leading to non-deterministic behaviour across test orders.

    Strategy: snapshot the set of ``sam3.*`` keys before each test and restore
    it after, so that any additions or deletions made during a test are rolled
    back.  This is safe because the module objects are not invalidated (they
    remain on the heap as long as other references exist); we are only restoring
    the mapping in sys.modules.

    Note: this does NOT prevent test_image_encoder_onnx.py from doing its own
    evict+reload inside the test body -- it restores the *snapshot* afterwards.
    """
    snapshot_keys = frozenset(k for k in sys.modules if k == "sam3" or k.startswith("sam3."))
    snapshot = {k: sys.modules[k] for k in snapshot_keys}

    yield

    # After the test: restore sam3.* to the pre-test state.
    # 1. Remove entries that were added during the test.
    for k in list(sys.modules):
        if (k == "sam3" or k.startswith("sam3.")) and k not in snapshot_keys:
            del sys.modules[k]
    # 2. Re-add entries that were removed during the test (e.g. by _evict_sam3_cache).
    for k, mod in snapshot.items():
        if k not in sys.modules:
            sys.modules[k] = mod
