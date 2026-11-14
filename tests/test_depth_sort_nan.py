"""Test case for the NaN depth-sorting bug (see issue #25 and PR #44).

Run it with pytest:

    pytest tests/test_depth_sort_nan.py -v

or directly, without pytest installed:

    python tests/test_depth_sort_nan.py

What this file proves
---------------------
The root cause is that `np.median()` returns NaN when its input is empty
(a fully transparent part crop makes the mask empty), and **a NaN sort key
breaks `list.sort()`**: comparisons against NaN are always False, so the
result stops depending on the key values and starts depending on the order
of the input list instead.

That gives a sharp, order-independent way to test the bug: sort the *same*
set of parts in different input orders and compare the results.

* without the fix: the resulting layer order **changes with the input order**
  (the bug -- layers shuffle around for no reason)
* with the fix: the resulting layer order is **always the same**, and it is
  the correct order by depth

This is a plain unit test: it needs numpy only, no model weights, no GPU.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

# The package is imported as `utils.*` in this repository (see
# `inference_utils.py`), so make `common/` importable when running
# this file directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _candidate in (_REPO_ROOT / "common", _REPO_ROOT):
    if (_candidate / "utils" / "depth_sort.py").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from utils.depth_sort import FAR_DEPTH, depth_sort_key, median_depth  # noqa: E402


# --------------------------------------------------------------------------- #
# the root cause: np.median() over an empty selection is NaN
# --------------------------------------------------------------------------- #
def test_empty_selection_gives_nan_without_help():
    """This is what the code did before the fix -- and why the bug happened."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # np.median([]) also warns
        assert np.isnan(np.median(np.array([])))


def test_median_depth_falls_back_instead_of_returning_nan():
    """`median_depth` must never hand a NaN to the sort."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert median_depth([]) == FAR_DEPTH
        assert median_depth(np.array([])) == FAR_DEPTH
    assert not [w for w in caught if "median" in str(w.message).lower()]


def test_median_depth_keeps_ordinary_values():
    assert median_depth([1.0, 2.0, 3.0]) == 2.0
    assert median_depth([0.5, 0.5, 0.5]) == 0.5
    assert median_depth([1.0, 2.0, float("nan")]) == FAR_DEPTH   # poisoned input


# --------------------------------------------------------------------------- #
# the symptom: the layer order must not depend on the input order
# --------------------------------------------------------------------------- #
def _parts_with_one_nan():
    """Four parts, one of them with an unknown (NaN) depth.

    `names` are the layer labels; the depth values are the ones a real run
    produced when one part's mask came out empty.
    """
    return [
        {"name": "face", "depth_median": 0.2},
        {"name": "body", "depth_median": 0.5},
        {"name": "hair", "depth_median": float("nan")},
        {"name": "bg", "depth_median": 0.9},
    ]


def _orders_with(parts, key):
    """Layer order for every permutation of the input list -> a set of results."""
    results = set()
    for perm in itertools.permutations(parts):
        ordered = sorted(perm, key=key)
        results.add(tuple(p["name"] for p in ordered))
    return results


def test_bug_reproduction_sort_order_depends_on_input_order():
    """Without the fix (plain `depth_median` key) the order is not well defined.

    This is the bug from issue #25: sorting silently misbehaves because a NaN
    is in the key, so identical data can come out in different layer orders.

    Note on the assertion: the exact number of different orders that come out
    is a CPython implementation detail and can differ between Python versions
    and platforms, so this only asserts the meaningful part -- that the unfixed
    sort does **not** reliably produce the correct order. That holds whichever
    order the interpreter happens to pick, which keeps the test stable while
    still failing on the unfixed code.
    """
    parts = _parts_with_one_nan()
    correct = ("face", "body", "bg", "hair")
    unstable = _orders_with(parts, key=lambda p: p["depth_median"])
    assert unstable != {correct}, (
        "the unfixed sort produced the correct order for every input order -- "
        "either the interpreter changed or the bug moved; check manually. "
        f"got: {sorted(unstable)}"
    )


def test_fixed_sort_is_order_independent_and_correct():
    """With the fix the order is always the same, and it is the depth order."""
    parts = _parts_with_one_nan()
    stable = _orders_with(parts, key=depth_sort_key)
    assert len(stable) == 1, f"sort order still depends on input order: {stable}"
    assert stable == {("face", "body", "bg", "hair")}, (
        "expected nearest-to-farthest with the unknown depth last, got "
        f"{stable}"
    )


# --------------------------------------------------------------------------- #
# the two call sites in this repository
# --------------------------------------------------------------------------- #
def test_hair_split_picks_front_and_back_correctly():
    """`parts.sort(key=lambda x: x['depth_median'])` in inference_utils.

    The hair split takes parts[0] as the front hair and parts[1] as the back
    hair, so a scrambled order swaps them.
    """
    split = [{"name": "back", "depth_median": 0.7},
             {"name": "front", "depth_median": float("nan")}]
    correct = ("back", "front")
    before = {tuple(p["name"] for p in sorted(perm, key=lambda p: p["depth_median"]))
              for perm in itertools.permutations(split)}
    after = {tuple(p["name"] for p in sorted(perm, key=depth_sort_key))
             for perm in itertools.permutations(split)}
    assert before != {correct}, (
        "the unfixed hair split gave the correct front/back for every input "
        f"order, which should not happen; got {sorted(before)}"
    )
    assert after == {correct}, after


def test_depth_sort_key_edge_cases():
    assert depth_sort_key({}) == FAR_DEPTH
    assert depth_sort_key({"depth_median": None}) == FAR_DEPTH
    assert depth_sort_key({"depth_median": "0.3"}) == 0.3
    assert depth_sort_key({"depth_median": float("inf")}) == FAR_DEPTH
    assert depth_sort_key({"depth_median": float("-inf")}) == FAR_DEPTH
    assert depth_sort_key({"depth_median": 0.0}) == 0.0


# --------------------------------------------------------------------------- #
# how the real code path produced the NaN (for reproducibility)
# --------------------------------------------------------------------------- #
def test_the_real_code_path_that_produces_nan():
    """`_compute_depth_median` uses `np.median(depth[mask])`; a fully
    transparent crop (an empty mask) is what makes that NaN.

    This mirrors the real computation without importing torch/cv2:
    an empty mask selects zero pixels, so the median is taken over nothing.
    """
    depth = np.array([[0.1, 0.2], [0.3, 0.4]])
    mask = np.array([[False, False], [False, False]])       # fully transparent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buggy = float(np.median(depth[mask])) if np.any(mask) else float("nan")
    assert np.isnan(buggy), "an empty mask is what produced the NaN"
    assert median_depth(depth[mask]) == FAR_DEPTH, "the fix must catch exactly this"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
