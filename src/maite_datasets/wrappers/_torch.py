from __future__ import annotations

from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar, cast, overload

import torch
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata
from maite.protocols.object_detection import ObjectDetectionTarget
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, Image

from maite_datasets._base import BaseDataset, ObjectDetectionTargetTuple
from maite_datasets.protocols import Array

TArray = TypeVar("TArray", bound=Array)
TTarget = TypeVar("TTarget")


def to_tensor(array: ArrayLike, dtype: torch.dtype | None = None) -> torch.Tensor:
    return (
        array.detach().clone().to(device="cpu", dtype=dtype)
        if isinstance(array, torch.Tensor)
        else torch.tensor(array, dtype=dtype)
    )


class TorchvisionObjectDetectionTarget(Protocol):
    @property
    def boxes(self) -> BoundingBoxes: ...
    @property
    def labels(self) -> Tensor: ...
    @property
    def scores(self) -> Tensor: ...


TorchvisionImageClassificationDatum: TypeAlias = tuple[Image, Tensor, DatumMetadata]
TorchvisionObjectDetectionDatum: TypeAlias = tuple[Image, ObjectDetectionTargetTuple, DatumMetadata]


class TorchvisionWrapper(Generic[TArray, TTarget]):
    """
    Lightweight wrapper converting numpy-based datasets to Torchvision tensors.

    Converts images to tv_tensor.Image and targets to the appropriate torchvision format.

    Parameters
    ----------
    dataset : Dataset
        Source dataset with numpy arrays
    transforms : callable, optional
        Torchvision v2 transform functions for targets
    """

    def __init__(
        self,
        dataset: BaseDataset[TArray, TTarget],
        transforms: Callable[[Any], Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms
        self.metadata: DatasetMetadata = {
            "id": f"TorchvisionWrapper({dataset.metadata['id']})",
            "index2label": dataset.metadata.get("index2label", {}),
        }

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to wrapped dataset."""
        return getattr(self._dataset, name)

    def __dir__(self) -> list[str]:
        """Include wrapped dataset attributes in dir() for IDE support."""
        wrapper_attrs = set(super().__dir__())
        dataset_attrs = set(dir(self._dataset))
        return sorted(wrapper_attrs | dataset_attrs)

    def _transform(self, datum: Any) -> Any:
        return self.transforms(datum) if self.transforms else datum

    @overload
    def __getitem__(self: TorchvisionWrapper[TArray, TArray], index: int) -> tuple[Image, Tensor, DatumMetadata]: ...
    @overload
    def __getitem__(
        self: TorchvisionWrapper[TArray, TTarget], index: int
    ) -> tuple[Image, TorchvisionObjectDetectionTarget, DatumMetadata]: ...

    def __getitem__(self, index: int) -> tuple[Image, Tensor | TorchvisionObjectDetectionTarget, DatumMetadata]:
        """Get item with torch tensor conversion."""
        image, target, metadata = self._dataset[index]

        # Convert image to torch tensor
        torch_image = Image(to_tensor(image))

        # Handle different target types
        if isinstance(target, Array):
            # Image classification case
            torch_target = to_tensor(target, dtype=torch.float32)
            torch_datum = self._transform((torch_image, torch_target, metadata))
            return cast(TorchvisionImageClassificationDatum, torch_datum)

        if isinstance(target, ObjectDetectionTarget):
            # Object detection case
            torch_boxes = BoundingBoxes(
                to_tensor(target.boxes), format="XYXY", canvas_size=(torch_image.shape[-2], torch_image.shape[-1])
            )  # type: ignore
            torch_labels = to_tensor(target.labels, dtype=torch.int64)
            torch_scores = to_tensor(target.scores, dtype=torch.float32)
            torch_target = ObjectDetectionTargetTuple(torch_boxes, torch_labels, torch_scores)
            torch_datum = self._transform((torch_image, torch_target, metadata))
            return cast(TorchvisionObjectDetectionDatum, torch_datum)

        raise TypeError(f"Unsupported target type: {type(target)}")

    def __str__(self) -> str:
        """String representation showing torch version."""
        nt = "\n    "
        base_name = f"{self._dataset.__class__.__name__.replace('Dataset', '')} Dataset"
        title = f"Torchvision Wrapped {base_name}" if not base_name.startswith("Torchvision") else base_name
        sep = "-" * len(title)
        attrs = [
            f"{' '.join(w.capitalize() for w in k.split('_'))}: {v}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        ]
        wrapped = f"{title}\n{sep}{nt}{nt.join(attrs)}"
        return f"{wrapped}\n\n{self._dataset}"

    def __len__(self) -> int:
        return self._dataset.__len__()
