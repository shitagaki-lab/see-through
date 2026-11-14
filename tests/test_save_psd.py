"""Regression tests for save_psd canvas dimensions (issue #46).

psd-tools takes size as (width, height); save_psd used to pass (h, w),
inverting non-square canvases. These tests save a PSD through save_psd
and read the canvas size back with psd-tools.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from utils.io_utils import save_psd


def _rgba_layer(height, width):
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[:, :, 3] = 255
    return layer


def test_save_psd_non_square_canvas_size(tmp_path):
    out = tmp_path / "out.psd"
    save_psd(str(out), [{"img": _rgba_layer(60, 80), "layer_name": "test"}],
             h=300, w=500)
    psd = PSDImage.open(out)
    assert psd.width == 500
    assert psd.height == 300


def test_save_psd_square_canvas_size(tmp_path):
    out = tmp_path / "out.psd"
    save_psd(str(out), [{"img": _rgba_layer(60, 80), "layer_name": "test"}],
             h=400, w=400)
    psd = PSDImage.open(out)
    assert psd.width == 400
    assert psd.height == 400
