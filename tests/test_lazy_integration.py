from __future__ import annotations

import numpy as np
import pytest

from maite_datasets._fileio import ResourcePart, URLResource
from maite_datasets._lazy import LazyArray
from maite_datasets.image_classification._mnist import MNIST
from maite_datasets.image_classification._ships import Ships
from maite_datasets.object_detection._milco import MILCO


@pytest.fixture
def ships(ship_fake):
    return Ships(root=ship_fake)


@pytest.fixture
def milco(milco_fake):
    return MILCO(root=milco_fake, image_set="train")


class TestLazyICDataset:
    def test_eager_default_returns_ndarray(self, ships):
        img, _, _ = ships[0]
        assert isinstance(img, np.ndarray)

    def test_lazy_returns_lazyarray(self, ships):
        ships.lazy = True
        img, _, _ = ships[0]
        assert isinstance(img, LazyArray)

    def test_lazy_shape_no_decode(self, ships, monkeypatch):
        ships.lazy = True
        decoded = {"n": 0}
        orig = ships._read_file

        def spy(path):
            decoded["n"] += 1
            return orig(path)

        monkeypatch.setattr(ships, "_read_file", spy)
        img, _, _ = ships[0]
        assert img.shape == (3, 10, 10)
        assert decoded["n"] == 0

    def test_lazy_materialize_matches_eager(self, ships):
        eager_img, _, _ = ships[0]
        ships.lazy = True
        lazy_img, _, _ = ships[0]
        np.testing.assert_array_equal(np.asarray(lazy_img), eager_img)

    def test_lazy_with_image_only_transform_defers(self, ship_fake, monkeypatch):
        ds = Ships(root=ship_fake, transforms=lambda img: img + 1)
        ds.lazy = True
        decoded = {"n": 0}
        orig = ds._read_file

        def spy(path):
            decoded["n"] += 1
            return orig(path)

        monkeypatch.setattr(ds, "_read_file", spy)
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)
        assert decoded["n"] == 0
        # Materializing applies the transform
        eager = Ships(root=ship_fake, transforms=lambda i: i + 1)
        eager_img, _, _ = eager[0]
        np.testing.assert_array_equal(np.asarray(img), eager_img)

    def test_lazy_with_tuple_transform_warns(self, ship_fake):
        def tuple_xform(datum: tuple) -> tuple:
            img, tgt, md = datum
            return img, tgt, md

        ds = Ships(root=ship_fake, transforms=tuple_xform)
        with pytest.warns(UserWarning, match="tuple-style"):
            ds.lazy = True

    def test_lazy_with_tuple_transform_passes_ndarray(self, ship_fake):
        """Tuple transforms must receive a materialized ndarray, not a LazyArray."""
        received: dict = {}

        def tuple_xform(datum: tuple) -> tuple:
            img, tgt, md = datum
            received["dtype"] = img.dtype
            received["ndim"] = img.ndim
            return img.astype(np.float32), tgt, md

        ds = Ships(root=ship_fake, transforms=tuple_xform)
        with pytest.warns(UserWarning):
            ds.lazy = True
        img, _, _ = ds[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.float32
        assert received["ndim"] == 3

    def test_lazy_with_mixed_transforms_no_double_apply(self, ship_fake):
        """Image-only transforms must run once even when tuple transforms force materialization."""

        def tuple_xform(datum: tuple) -> tuple:
            img, tgt, md = datum
            return img * 2, tgt, md

        ds_eager = Ships(root=ship_fake, transforms=[lambda i: i + 1, tuple_xform])
        eager_img, _, _ = ds_eager[0]

        ds_lazy = Ships(root=ship_fake, transforms=[lambda i: i + 1, tuple_xform])
        with pytest.warns(UserWarning):
            ds_lazy.lazy = True
        lazy_img, _, _ = ds_lazy[0]
        np.testing.assert_array_equal(lazy_img, eager_img)


class TestLazyODDataset:
    def test_lazy_returns_lazyarray(self, milco):
        milco.lazy = True
        img, _, _ = milco[0]
        assert isinstance(img, LazyArray)

    def test_lazy_bboxes_per_size_uses_lazy_shape(self, milco, monkeypatch):
        milco.lazy = True
        decoded = {"n": 0}
        orig = milco._read_file

        def spy(path):
            decoded["n"] += 1
            return orig(path)

        monkeypatch.setattr(milco, "_read_file", spy)
        img, target, _ = milco[0]
        # Box scaling reads shape; should NOT decode pixels.
        assert isinstance(img, LazyArray)
        assert decoded["n"] == 0
        # Target boxes were scaled by image dimensions — must be finite.
        assert np.all(np.isfinite(np.asarray(target.boxes)))

    def test_lazy_works_on_inmemory_dataset_via_fallback(self, mnist_npy, monkeypatch):
        """MNIST uses int-indexed string paths; _read_shape must fall back to _read_file."""
        import hashlib

        def get_hash(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65535):
                    h.update(chunk)
            return h.hexdigest()

        monkeypatch.setattr(
            MNIST,
            "_resources",
            [
                ResourcePart(
                    "mnist",
                    (
                        URLResource(
                            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                            filename="mnist.npz",
                            md5=False,
                            checksum=get_hash(mnist_npy / "mnist.npz"),
                        ),
                    ),
                ),
            ],
        )
        ds = MNIST(root=mnist_npy, download=False)
        ds.lazy = True
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)
        # Shape must work despite "path" being an integer string like "0".
        assert img.shape == (1, 28, 28)
        # Materialization works too.
        assert np.asarray(img).shape == (1, 28, 28)

    def test_lazy_target_matches_eager(self, milco_fake):
        eager = MILCO(root=milco_fake, image_set="train")
        eager_img, eager_tgt, eager_md = eager[0]
        lazy_ds = MILCO(root=milco_fake, image_set="train")
        lazy_ds.lazy = True
        lazy_img, lazy_tgt, lazy_md = lazy_ds[0]
        np.testing.assert_array_equal(np.asarray(lazy_img), eager_img)
        np.testing.assert_array_equal(np.asarray(lazy_tgt.boxes), np.asarray(eager_tgt.boxes))
        np.testing.assert_array_equal(np.asarray(lazy_tgt.labels), np.asarray(eager_tgt.labels))


class TestLazyConstructorParam:
    """The ``lazy`` constructor kwarg must set the property and yield LazyArrays."""

    def test_ships_constructor_lazy(self, ship_fake):
        ds = Ships(root=ship_fake, lazy=True)
        assert ds.lazy is True
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)

    def test_ships_constructor_default_eager(self, ship_fake):
        ds = Ships(root=ship_fake)
        assert ds.lazy is False
        img, _, _ = ds[0]
        assert isinstance(img, np.ndarray)

    def test_milco_constructor_lazy(self, milco_fake):
        ds = MILCO(root=milco_fake, image_set="train", lazy=True)
        assert ds.lazy is True
        img, _, _ = ds[0]
        assert isinstance(img, LazyArray)
