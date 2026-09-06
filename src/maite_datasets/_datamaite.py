"""Integration bridge between maite-datasets and datamaite."""

from __future__ import annotations

import inspect
import io
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Splits the datamaite YOLO writers round-trip. A directory named after anything else
# is written happily and then skipped by the loader, so every other ``image_set``
# (``operational``, ``base``, ...) has to be folded into one of these.
_ROUND_TRIPPABLE_SPLITS = frozenset({"train", "val", "test"})

# Suffixes we are willing to hand to datamaite as an image reference. Datasets that
# load from a single archive (MNIST) put non-paths in ``_filepaths``, so matching an
# image extension -- not just "the path exists" -- is what keeps those off this path.
_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def check_datamaite_available() -> None:
    """Verify that datamaite is installed; raise ImportError with installation hint if not."""
    try:
        import datamaite  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The `as_datamaite` option requires datamaite to be installed. "
            "Install it with `pip install maite-datasets[datamaite]` or `pip install datamaite`."
        ) from err


def get_dataset_task(cls: type) -> str:
    """Determine whether a dataset class is for object detection ('od') or image classification ('ic')."""
    from maite_datasets._base import BaseICDataset, BaseODDataset
    from maite_datasets.object_detection._yolo import YOLODataset

    if issubclass(cls, (BaseODDataset, YOLODataset)):
        return "od"
    if issubclass(cls, BaseICDataset):
        return "ic"
    mod = cls.__module__
    if "object_detection" in mod:
        return "od"
    if "image_classification" in mod:
        return "ic"
    raise ValueError(f"Cannot determine vision task for {cls.__name__}")


def format_for_task(task: str) -> str:
    """The wire format an export of `task` is written in.

    One format per task rather than a choice, because each task only has one that works.
    Object detection writes COCO: it is the only one of the two that records a datum's own
    metadata -- a dataset's telemetry as per-image columns, its per-object values as
    detection attributes -- which is what the analyses reading these exports consume.
    Image classification writes YOLO because datamaite's COCO writer is task-closed and
    refuses an IC dataset outright.

    Reading stays more permissive: :func:`detect_format` sniffs either format, so an
    export or a hand-built directory already on disk loads whichever way it was written.
    """
    return "yolo" if task == "ic" else "coco"


def detect_format(dest: Path, task: str) -> str | None:
    """Detect if `dest` holds a valid datamaite dataset format using datamaite's sniffers."""
    if not dest.is_dir() or not any(dest.iterdir()):
        return None

    if task == "od":
        from datamaite._formats.coco.loader import CocoLoader
        from datamaite._formats.yolo.loader import YoloObjectDetectionLoader

        if CocoLoader.sniff(dest):
            return "coco"
        if YoloObjectDetectionLoader.sniff(dest):
            return "yolo"
        return None
    if task == "ic":
        from datamaite._formats.yolo.loader import YoloImageClassificationLoader

        if YoloImageClassificationLoader.sniff(dest):
            return "yolo"
        return None
    return None


def _resolve_dataset_dir(root: Path, name: str) -> Path:
    """Destination for a dataset's datamaite export under `root`.

    Deliberately not the ``root / name.lower()`` folder raw downloads use: sharing it
    would make a root that already holds a normal download unusable for ``as_datamaite``.
    ``root`` is used as-is when it already points at the export folder.
    """
    dirname = f"{name.lower()}_datamaite"
    return root if root.stem.lower() == dirname else root / dirname


def load_datamaite_dataset(dest: Path, task: str, dataset_format: str, split: str | None = None, **options: Any) -> Any:
    """Load an existing datamaite dataset from disk."""
    import datamaite

    if task == "od":
        load_kwargs: dict[str, Any] = {"dataset_format": dataset_format}
        if dataset_format == "yolo":
            load_kwargs["split"] = _normalize_split(split)
        load_kwargs.update(options)
        return datamaite.load_od(dest, **load_kwargs)
    if task == "ic":
        load_kwargs = {"dataset_format": dataset_format}
        if dataset_format == "yolo":
            load_kwargs["split"] = _normalize_split(split)
        load_kwargs.update(options)
        return datamaite.load_ic(dest, **load_kwargs)
    raise ValueError(f"Unsupported task: {task}")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale pixel values into uint8 without flattening float imagery to black."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        # Floats are conventionally normalized to [0, 1]; a plain cast would truncate
        # every one of those pixels to 0. Wider float ranges are already 8-bit scaled.
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_pil_layout(arr: np.ndarray) -> np.ndarray:
    """Reshape a MAITE (C, H, W) image into a channel layout PIL can encode."""
    if arr.ndim != 3:
        return arr
    arr = np.transpose(arr, (1, 2, 0))
    channels = arr.shape[2]
    if channels == 1:
        return arr.squeeze(2)
    if channels in (3, 4):
        return arr
    # PNG has no mode for 2- or 5+-channel imagery. Keeping the leading band(s) is the
    # only encodable option, so say so rather than dropping data silently.
    kept = "first channel" if channels < 3 else "first three channels"
    warnings.warn(
        f"Cannot encode a {channels}-channel image as PNG; exporting the {kept}.",
        UserWarning,
        stacklevel=3,
    )
    return arr[:, :, 0] if channels < 3 else arr[:, :, :3]


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert an image numpy array to PNG bytes."""
    pil_img = Image.fromarray(_to_pil_layout(_to_uint8(np.asarray(arr))))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _index2label(raw_ds: Any) -> dict[int, str]:
    """Class map for either dataset hierarchy.

    ``BaseDownloadedDataset`` publishes ``index2label``; ``BaseReaderDataset`` calls the
    same mapping ``classes``. Both populate ``metadata["index2label"]``, so that is the
    one lookup that works for every dataset.
    """
    index2label = raw_ds.metadata.get("index2label") if hasattr(raw_ds, "metadata") else None
    if not index2label:
        index2label = getattr(raw_ds, "index2label", None) or getattr(raw_ds, "classes", None)
    return dict(index2label or {})


def _source_image_path(raw_ds: Any, index: int, datum_meta: Any) -> Path | None:
    """Path of the on-disk image backing datum `index`, or None when there isn't one.

    Reader-backed datasets have no ``_filepaths`` and publish the path per datum
    instead; array-backed datasets (MNIST) put indices in ``_filepaths``, which is why
    the result has to look like an actual image file before we hand it to datamaite.
    """
    candidate = datum_meta.get("file_path") if hasattr(datum_meta, "get") else None
    if not candidate:
        filepaths = getattr(raw_ds, "_filepaths", None)
        if filepaths is not None and index < len(filepaths):
            candidate = filepaths[index]
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file() else None


@contextmanager
def _prepared_for_export(raw_ds: Any) -> Iterator[bool]:
    """Set up `raw_ds` for a conversion pass and yield whether source files may be referenced.

    A transformed datum no longer matches the file it came from, so transformed datasets
    export re-encoded pixels; untransformed ones reference the original file and never
    need a decode at all, which is what ``lazy`` buys us here. Either way ``lazy`` is the
    caller's setting, so it is restored on the way out.
    """
    can_reference_source = not getattr(raw_ds, "transforms", None)
    original_lazy = raw_ds.lazy
    if can_reference_source:
        raw_ds.lazy = True
    try:
        yield can_reference_source
    finally:
        raw_ds.lazy = original_lazy


def _image_payload(
    raw_ds: Any, index: int, img: Any, datum_meta: Any, can_reference_source: bool
) -> tuple[str | None, bytes | None, str]:
    """Return the (path_or_uri, image_bytes, file_name) triple describing one datum's pixels."""
    source = _source_image_path(raw_ds, index, datum_meta) if can_reference_source else None
    if source is not None:
        return str(source), None, source.name
    return None, _array_to_png_bytes(np.asarray(img)), f"{index:06d}.png"


# Sentinel for "this metadata value is not a scalar", so a legitimate None still passes.
_NOT_SCALAR = object()


def _as_scalar(value: Any) -> Any:
    """`value` as a JSON-safe scalar, or `_NOT_SCALAR` when it is not one."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return _NOT_SCALAR


def _split_datum_metadata(datum_meta: Any, num_boxes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sort per-datum metadata into per-image values and per-object values.

    A datum's metadata mixes the two: SeaDrone reports one ``altitude`` for the frame and
    one ``object_id`` per annotated object. Publishing the per-object lists as image
    metadata would make each an unreadable column, and dropping them loses real data, so
    a sequence whose length matches the boxes is attributed to them one by one. Anything
    that is neither a scalar nor an aligned sequence cannot be placed and is left out.
    """
    per_image: dict[str, Any] = {}
    per_object: list[dict[str, Any]] = [{} for _ in range(num_boxes)]
    for key, value in datum_meta.items():
        if key in ("id", "original_id", "original_split"):
            continue
        scalar = _as_scalar(value)
        if scalar is not _NOT_SCALAR:
            per_image[str(key)] = scalar
            continue
        values = _aligned_sequence(value, num_boxes)
        if values is not None:
            for attributes, item in zip(per_object, values):
                attributes[str(key)] = item
    return per_image, per_object


def _aligned_sequence(value: Any, num_boxes: int) -> list[Any] | None:
    """`value` as one scalar per box, or None when it does not line up with them."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        return None
    if len(value) != num_boxes:
        return None
    items = [_as_scalar(item) for item in value]
    return None if any(item is _NOT_SCALAR for item in items) else items


def _provenance(raw_id: Any, image_id: Any, split: str | None, effective_split: str) -> dict[str, Any]:
    """Identity the exported sample cannot carry on its own.

    Both keys become ordinary columns to whatever reads the export, so each is recorded
    only when it actually adds something: ``original_id`` when the id could not be kept
    (COCO image ids are integers, so a string id has to go somewhere), ``original_split``
    when the split was folded. Emitting them unconditionally would hand metadata analysis
    a duplicate of ``image_id`` and of ``sample.split`` to trip over.
    """
    provenance: dict[str, Any] = {}
    if str(raw_id) != str(image_id):
        provenance["original_id"] = str(raw_id)
    if split and split != effective_split:
        provenance["original_split"] = str(split)
    return provenance


def _coco_image_id(raw_id: Any, index: int, taken: set[int]) -> int:
    """A unique integer image id for `raw_id`, falling back to the datum's position.

    Only an id that is already an integer is reused. ``int()`` would otherwise read a
    stem like ``"1_2015"`` as 12015 -- Python's digit separators quietly turning two
    unrelated ids into one number -- so anything else is numbered by position instead
    and keeps its real identity in the sample's ``original_id`` metadata.
    """
    if isinstance(raw_id, (int, np.integer)) and not isinstance(raw_id, bool):
        candidate = int(raw_id)
    elif isinstance(raw_id, str) and raw_id.strip().lstrip("-").isdigit():
        candidate = int(raw_id.strip())
    else:
        candidate = index + 1
    while candidate in taken:
        candidate += 1
    return candidate


def _normalize_split(split: str | None) -> str:
    """Fold a maite-datasets `image_set` onto a split name datamaite can read back."""
    return split if split in _ROUND_TRIPPABLE_SPLITS else "train"


def _export_split(split: str | None) -> str:
    """`_normalize_split` for a conversion, announcing a fold the export cannot record.

    ``base`` and an unset split already mean "no particular split", so only a named one
    (``operational``) is worth reporting -- writing it verbatim would produce a split
    directory the loaders skip, which is the silent-data-loss case this avoids.
    """
    normalized = _normalize_split(split)
    if split and split != normalized and split != "base":
        warnings.warn(
            f"datamaite reads back only {sorted(_ROUND_TRIPPABLE_SPLITS)} splits; "
            f"exporting image_set '{split}' as '{normalized}'.",
            UserWarning,
            stacklevel=3,
        )
    return normalized


def _extract_score(scores: Any, b_idx: int, lbl: int) -> float:
    """Extract a scalar detection score from 1D, 2D (one-hot), or absent score arrays."""
    if scores is None or len(scores) <= b_idx:
        return 1.0
    s = np.asarray(scores[b_idx])
    if s.ndim > 0:
        return float(s[lbl]) if lbl < len(s) else float(np.max(s))
    return float(s)


def _convert_od_dataset(raw_ds: Any, split: str | None = None) -> Any:
    """Convert a MAITE OD dataset instance to datamaite.ObjectDetectionDataset."""
    from datamaite.object_detection import ObjectDetectionDataset
    from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
    from datamaite.taxonomy import CategoryEntry, Taxonomy

    samples = []
    seen_ids: set[int] = set()
    num_items = len(raw_ds)
    index2label = _index2label(raw_ds)
    effective_split = _export_split(split)

    with _prepared_for_export(raw_ds) as can_reference_source:
        for i in range(num_items):
            img, target, datum_meta = raw_ds[i]
            h = int(img.shape[1])
            w = int(img.shape[2])

            path_or_uri, image_bytes, file_name = _image_payload(raw_ds, i, img, datum_meta, can_reference_source)

            detections = []
            boxes = target.boxes
            labels = target.labels
            scores = getattr(target, "scores", None)
            image_metadata, object_metadata = _split_datum_metadata(datum_meta, len(boxes))

            for b_idx in range(len(boxes)):
                box = np.asarray(boxes[b_idx]).flatten()
                x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bx = x0
                by = y0
                bw = max(0.0, x1 - x0)
                bh = max(0.0, y1 - y0)

                lbl = int(np.asarray(labels[b_idx]).item())
                cat_name = index2label.get(lbl, str(lbl))
                sc = _extract_score(scores, b_idx, lbl)

                detections.append(
                    ObjectDetectionAnnotation(
                        bbox=(bx, by, bw, bh),
                        category_id=lbl,
                        category_name=cat_name,
                        score=sc,
                        attributes=object_metadata[b_idx],
                    )
                )

            raw_id = datum_meta.get("id", i)
            image_id = _coco_image_id(raw_id, i, seen_ids)
            seen_ids.add(image_id)

            samples.append(
                ImageObjectDetectionSample(
                    image_id=image_id,
                    path_or_uri=path_or_uri,
                    image_bytes=image_bytes,
                    file_name=file_name,
                    width=w,
                    height=h,
                    split=effective_split,
                    metadata=image_metadata | _provenance(raw_id, image_id, split, effective_split),
                    detections=tuple(detections),
                )
            )

    entries = tuple(CategoryEntry(source_id=idx, name=str(name)) for idx, name in sorted(index2label.items()))
    taxonomy = Taxonomy(entries=entries, id_density="dense")
    meta = DatasetMetadata(
        taxonomy=taxonomy,
        splits=(effective_split,),
        source_dataset=raw_ds.__class__.__name__,
    )
    return ObjectDetectionDataset(
        samples=tuple(samples),
        dataset_metadata=meta,
        dataset_id=f"{raw_ds.__class__.__name__}_{effective_split}",
    )


def _convert_ic_dataset(raw_ds: Any, split: str | None = None) -> Any:
    """Convert a MAITE IC dataset instance to datamaite.ImageClassificationDataset."""
    from datamaite.image_classification import ImageClassificationDataset
    from datamaite.records import ClassificationLabel, DatasetMetadata, ImageClassificationSample
    from datamaite.taxonomy import CategoryEntry, Taxonomy

    samples = []
    num_items = len(raw_ds)
    index2label = _index2label(raw_ds)
    effective_split = _export_split(split)

    with _prepared_for_export(raw_ds) as can_reference_source:
        for i in range(num_items):
            img, target, datum_meta = raw_ds[i]
            if hasattr(raw_ds, "_targets"):
                label = int(np.asarray(raw_ds._targets[i]).item())
            else:
                label = int(np.asarray(np.argmax(target)).item())
            cat_name = index2label.get(label, str(label))
            img_id = datum_meta.get("id", i)

            path_or_uri, image_bytes, file_name = _image_payload(raw_ds, i, img, datum_meta, can_reference_source)

            samples.append(
                ImageClassificationSample(
                    image_id=str(img_id),
                    path_or_uri=path_or_uri,
                    image_bytes=image_bytes,
                    file_name=file_name,
                    split=effective_split,
                    metadata=_split_datum_metadata(datum_meta, 0)[0]
                    | _provenance(img_id, str(img_id), split, effective_split),
                    labels=(ClassificationLabel(category_id=label, category_name=cat_name),),
                )
            )

    entries = tuple(CategoryEntry(source_id=idx, name=str(name)) for idx, name in sorted(index2label.items()))
    taxonomy = Taxonomy(entries=entries, id_density="dense")
    meta = DatasetMetadata(
        taxonomy=taxonomy,
        splits=(effective_split,),
        source_dataset=raw_ds.__class__.__name__,
    )
    return ImageClassificationDataset(
        samples=tuple(samples),
        dataset_metadata=meta,
        dataset_id=f"{raw_ds.__class__.__name__}_{effective_split}",
    )


def write_as_datamaite(
    raw_ds: Any,
    dest: Path,
    task: str,
    split: str | None = None,
) -> None:
    """Convert and write a MAITE dataset instance into dest using datamaite.write."""
    import datamaite

    if task == "od":
        dm_ds = _convert_od_dataset(raw_ds, split=split)
    elif task == "ic":
        dm_ds = _convert_ic_dataset(raw_ds, split=split)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # "append" would make a second export of the same dataset duplicate every sample,
    # so an export always defines the whole contents of `dest`.
    datamaite.write(dm_ds, dest, output_format=format_for_task(task), mode="replace")


def _extract_parameters(
    cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Path, str, bool, dict[str, Any]]:
    """Extract root, image_set, download and the remaining kwargs from call arguments."""
    kwargs_copy = dict(kwargs)
    kwargs_copy.pop("as_datamaite", None)

    sig = inspect.signature(cls.__init__)
    bound = sig.bind_partial(None, *args, **kwargs_copy)
    bound.apply_defaults()

    root = bound.arguments.get("root")
    if root is None:
        raise ValueError(f"Missing required 'root' parameter for dataset {cls.__name__}.")

    image_set = bound.arguments.get("image_set", "train")
    download = bound.arguments.get("download", False)

    bound_args = dict(bound.arguments)
    bound_args.pop("self", None)
    bound_args.pop("root", None)
    bound_args.pop("download", None)
    bound_args.pop("as_datamaite", None)

    return Path(root), image_set, download, bound_args


@contextmanager
def _build_raw_dataset(cls: type, root: Path, download: bool, bound_kwargs: dict[str, Any]) -> Iterator[Any]:
    """Construct the plain dataset to convert, without leaving a raw copy behind.

    An already-downloaded `root` is read in place. Otherwise the raw download goes to a
    temporary directory that is discarded once converted, so ``as_datamaite`` never
    leaves the un-converted dataset sitting next to its export.
    """
    try:
        local = cls(**(dict(bound_kwargs) | {"root": root, "download": False, "as_datamaite": False}))
    except FileNotFoundError:
        local = None
    if local is not None:
        yield local
        return

    if not download:
        raise FileNotFoundError(
            f"Dataset not found at '{root}'. Set download=True to download and format as datamaite."
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield cls(**(dict(bound_kwargs) | {"root": Path(tmp_dir), "download": True, "as_datamaite": False}))


def build_datamaite_dataset(cls: type, *args: Any, **kwargs: Any) -> Any:
    """Intercept dataset construction when as_datamaite=True."""
    check_datamaite_available()

    root, image_set, download, bound_kwargs = _extract_parameters(cls, args, kwargs)
    task = get_dataset_task(cls)
    dest = _resolve_dataset_dir(root, cls.__name__)

    # 1. A previous export is reused as-is.
    if dest.exists() and any(dest.iterdir()):
        detected_fmt = detect_format(dest, task)
        if detected_fmt is None:
            raise ValueError(
                f"Dataset directory '{dest}' already exists but is not in a datamaite-compatible format. "
                "Please remove it or choose a different root directory."
            )
        return load_datamaite_dataset(dest, task=task, dataset_format=detected_fmt, split=image_set)

    # 2. Otherwise convert the raw dataset -- downloading it first only if it is missing.
    with _build_raw_dataset(cls, root, download, bound_kwargs) as raw_ds:
        write_as_datamaite(raw_ds, dest, task=task, split=image_set)

    return load_datamaite_dataset(dest, task=task, dataset_format=format_for_task(task), split=image_set)


def export_to_datamaite(dataset: Any, dest: str | Path) -> Any:
    """Export a MAITE dataset instance to disk in a datamaite-compatible format."""
    check_datamaite_available()
    dest = Path(dest).absolute()
    task = get_dataset_task(dataset.__class__)
    dataset_format = format_for_task(task)
    image_set = getattr(dataset, "image_set", "train")

    write_as_datamaite(dataset, dest, task=task, split=image_set)
    return load_datamaite_dataset(dest, task=task, dataset_format=dataset_format, split=image_set)
