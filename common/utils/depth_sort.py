"""NaN-safe helpers for part depth values.

`depth_median` is used as a sort key when the parts are turned into PSD layers
(see `inference_utils.dump_parts_psd` and the hair split in `inference_utils`).

Two things can make that value NaN:

1. the mask used to select the pixels of a part is empty
   -> `np.median([])` is NaN, and
2. the depth map itself can contain NaN for degenerate crops.

NaN breaks `list.sort()`: comparisons against NaN are always False, so the
result depends on the order of the input list instead of the key values.
For parts this shows up as a scrambled layer order (arms/hats/front and back
hair swapping places) even though nothing else changed.

`FAR_DEPTH` is the value this project already uses for "farthest / behind":
`inference_utils` initialises `depth_median = 1` and reads it back with
`.get('depth_median', 1)`. Unknown depths are therefore treated as farthest,
which keeps the previous default behaviour instead of inventing a new one.
"""
import numpy as np

#: Sentinel for "farthest / behind"; matches the existing defaults in
#: `inference_utils` (`depth_median = 1`).
FAR_DEPTH = 1.0


def median_depth(values, default=FAR_DEPTH):
    """Median of `values`, falling back to `default` when there is nothing to measure.

    `np.median([])` is NaN (plus a RuntimeWarning), and a NaN here would
    silently poison every later sort.
    """
    values = np.asarray(values)
    if values.size == 0:
        return default
    median = float(np.median(values))
    return default if not np.isfinite(median) else median


def depth_sort_key(part, default=FAR_DEPTH):
    """Sort key for a part dict: unknown / non-finite depths sort as farthest."""
    value = part.get('depth_median', default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(value) else value
