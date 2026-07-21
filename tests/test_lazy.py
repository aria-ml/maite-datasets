from __future__ import annotations

import numpy as np
import pytest
import tifffile as tif
from PIL import Image

from maite_datasets._lazy import LazyArray, chw_loaders, tiff_chw_load, tiff_chw_shape
from maite_datasets.protocols import Array


@pytest.fixture
def rgb_png(tmp_path):
    path = tmp_path / "img.png"
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    Image.fromarray(arr).save(path)
    return path


def _pil_shape(path):
    with Image.open(path) as im:
        mode_to_channels = {"L": 1, "RGB": 3, "RGBA": 4}
        c = mode_to_channels[im.mode]
        w, h = im.size
    return (c, h, w)


def _pil_load(path):
    return np.array(Image.open(path)).transpose(2, 0, 1)


class TestLazyArray:
    def test_satisfies_array_protocol(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        assert isinstance(lazy, Array)

    def test_shape_via_shape_loader_no_materialization(self, rgb_png):
        calls = {"load": 0, "shape": 0}

        def loader(p):
            calls["load"] += 1
            return _pil_load(p)

        def shape_loader(p):
            calls["shape"] += 1
            return _pil_shape(p)

        lazy = LazyArray(str(rgb_png), loader=loader, shape_loader=shape_loader)
        assert lazy.shape == (3, 2, 3)
        assert calls["load"] == 0
        assert calls["shape"] == 1

    def test_array_dunder_materializes(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        arr = np.asarray(lazy)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2, 3)
        np.testing.assert_array_equal(arr, _pil_load(str(rgb_png)))

    def test_materialize_caches(self, rgb_png):
        calls = {"load": 0}

        def loader(p):
            calls["load"] += 1
            return _pil_load(p)

        lazy = LazyArray(str(rgb_png), loader=loader, shape_loader=_pil_shape)
        np.asarray(lazy)
        np.asarray(lazy)
        assert calls["load"] == 1

    def test_getitem_materializes(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        np.testing.assert_array_equal(lazy[0], _pil_load(str(rgb_png))[0])

    def test_iter_materializes(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        rows = list(lazy)
        np.testing.assert_array_equal(np.stack(rows), _pil_load(str(rgb_png)))

    def test_len_uses_shape(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        assert len(lazy) == 3

    def test_no_shape_loader_falls_back_to_load(self, rgb_png):
        calls = {"load": 0}

        def loader(p):
            calls["load"] += 1
            return _pil_load(p)

        lazy = LazyArray(str(rgb_png), loader=loader)
        assert lazy.shape == (3, 2, 3)
        assert calls["load"] == 1

    def test_pending_transforms_applied_in_order_on_materialize(self, rgb_png):
        lazy = LazyArray(
            str(rgb_png),
            loader=_pil_load,
            shape_loader=_pil_shape,
            pending=[lambda a: a + 1, lambda a: a * 2],
        )
        arr = np.asarray(lazy)
        expected = (_pil_load(str(rgb_png)) + 1) * 2
        np.testing.assert_array_equal(arr, expected)

    def test_pending_transforms_force_shape_materialization(self, rgb_png):
        """Pending transforms can change shape, so shape must load."""
        calls = {"load": 0}

        def loader(p):
            calls["load"] += 1
            return _pil_load(p)

        lazy = LazyArray(
            str(rgb_png),
            loader=loader,
            shape_loader=_pil_shape,
            pending=[lambda a: a[:, :1, :]],
        )
        assert lazy.shape == (3, 1, 3)
        assert calls["load"] == 1

    def test_pending_metadata_only_iteration_skips_load(self, rgb_png):
        """No shape/array access with pending transforms: no decode."""
        calls = {"load": 0, "shape": 0}

        def loader(p):
            calls["load"] += 1
            return _pil_load(p)

        def shape_loader(p):
            calls["shape"] += 1
            return _pil_shape(p)

        LazyArray(str(rgb_png), loader=loader, shape_loader=shape_loader, pending=[lambda a: a + 1])
        assert calls["load"] == 0
        assert calls["shape"] == 0

    def test_array_dunder_respects_dtype(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        arr = np.asarray(lazy, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_repr_indicates_lazy_then_loaded(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        assert "lazy" in repr(lazy)
        np.asarray(lazy)
        assert "loaded" in repr(lazy)

    def test_array_copy_false_raises_on_dtype_cast(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        with pytest.raises(ValueError, match="copy"):
            np.asarray(lazy, dtype=np.float32, copy=False)

    def test_array_copy_false_returns_view_when_no_cast(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        arr = np.asarray(lazy, copy=False)
        # Underlying buffer is the cached array.
        np.asarray(lazy)[0, 0, 0] = 99
        assert arr[0, 0, 0] == 99

    def test_array_copy_true_does_not_alias(self, rgb_png):
        lazy = LazyArray(str(rgb_png), loader=_pil_load, shape_loader=_pil_shape)
        arr = np.array(lazy)  # default np.array copies
        np.asarray(lazy)[0, 0, 0] = 99
        assert arr[0, 0, 0] != 99


@pytest.fixture
def gray_tiff(tmp_path):
    path = tmp_path / "gray.tiff"
    tif.imwrite(path, np.arange(6 * 8, dtype=np.uint8).reshape(6, 8))
    return path


@pytest.fixture
def rgb_tiff(tmp_path):
    path = tmp_path / "rgb.tiff"
    tif.imwrite(path, np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3))
    return path


class TestTiffLoaders:
    def test_grayscale_gets_channel_axis(self, gray_tiff):
        assert tiff_chw_load(gray_tiff).shape == (1, 6, 8)
        assert tiff_chw_shape(gray_tiff) == (1, 6, 8)

    def test_sample_axis_moved_to_front(self, rgb_tiff):
        arr = tiff_chw_load(rgb_tiff)
        assert arr.shape == (3, 6, 8)
        assert tiff_chw_shape(rgb_tiff) == (3, 6, 8)
        # Same pixels, just reordered from the stored HWC layout
        assert np.array_equal(arr, np.moveaxis(tif.imread(rgb_tiff), 2, 0))

    @pytest.mark.parametrize("fixture, expected", [("gray_tiff", (1, 6, 8)), ("rgb_tiff", (3, 6, 8))])
    def test_chw_loaders_routes_tiffs(self, request, fixture, expected):
        path = request.getfixturevalue(fixture)
        loader, shape_loader = chw_loaders(path)
        assert loader is tiff_chw_load
        assert shape_loader(path) == expected
        assert loader(path).shape == expected

    def test_lazy_tiff_materializes_through_loaders(self, rgb_tiff):
        loader, shape_loader = chw_loaders(rgb_tiff)
        lazy = LazyArray(str(rgb_tiff), loader=loader, shape_loader=shape_loader)
        assert lazy.shape == (3, 6, 8)
        assert np.asarray(lazy).shape == (3, 6, 8)
