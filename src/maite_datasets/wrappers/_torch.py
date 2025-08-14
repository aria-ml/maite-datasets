from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, overload

import numpy as np
import torch
from torchvision.tv_tensors import BoundingBoxes, Image

from maite_datasets._base import BaseDataset, GenericObjectDetectionTarget
from maite_datasets._protocols import Array, DatasetMetadata, DatumMetadata, ObjectDetectionTarget

T = TypeVar("T")
TArray = TypeVar("TArray", bound=Array)
TTarget = TypeVar("TTarget")


class TorchWrapper(Generic[TArray, TTarget]):
    """
    Lightweight wrapper converting numpy-based datasets to PyTorch tensors.

    Converts images to torch.Tensor and targets to specified torch-compatible format.

    Parameters
    ----------
    dataset : Dataset
        Source dataset with numpy arrays
    transforms : callable, optional
        Torchvision transform function for targets
    """

    def __init__(
        self,
        dataset: BaseDataset[tuple[TArray, TTarget, DatumMetadata]],
        transforms: Callable[[Any], Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms
        self.metadata: DatasetMetadata = {
            "id": f"TorchWrapper({dataset.metadata['id']})",
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

    @overload
    def __getitem__(self: TorchWrapper[TArray, TArray], index: int) -> tuple[Image, torch.Tensor, DatumMetadata]: ...

    @overload
    def __getitem__(
        self: TorchWrapper[TArray, TTarget], index: int
    ) -> tuple[Image, GenericObjectDetectionTarget[torch.Tensor], DatumMetadata]: ...

    def __getitem__(
        self, index: int
    ) -> (
        tuple[Image, torch.Tensor, DatumMetadata]
        | tuple[Image, GenericObjectDetectionTarget[torch.Tensor], DatumMetadata]
    ):
        """Get item with torch tensor conversion."""
        image, target, metadata = self._dataset[index]

        # Convert image to torch tensor
        torch_image = torch.from_numpy(image) if isinstance(image, np.ndarray) else torch.as_tensor(image)
        if torch_image.dtype == torch.uint8:
            torch_image = torch_image.float() / 255.0
        torch_image = Image(torch_image)

        # Handle different target types
        if isinstance(target, Array):
            # Image classification case
            torch_target = torch.as_tensor(target, dtype=torch.float32)
            target_dict = {"labels": torch_target}
        elif isinstance(target, ObjectDetectionTarget):
            # Object detection case
            torch_boxes = BoundingBoxes(
                torch.as_tensor(target.boxes), format="XYXY", canvas_size=(torch_image.shape[-2], torch_image.shape[-1])
            )  # type: ignore
            torch_labels = torch.as_tensor(target.labels, dtype=torch.int64)
            torch_scores = torch.as_tensor(target.scores, dtype=torch.float32)
            target_dict = {"boxes": torch_boxes, "labels": torch_labels, "scores": torch_scores}
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

        if self.transforms:
            torch_image, target_dict = self.transforms(torch_image, target_dict)  # type: ignore

        # Return appropriate target type
        if isinstance(target, Array):
            return torch_image, target_dict["labels"], metadata
        return torch_image, GenericObjectDetectionTarget(**target_dict), metadata

    def __str__(self) -> str:
        """String representation showing torch version."""
        nt = "\n    "
        base_name = f"{self._dataset.__class__.__name__.replace('Dataset', '')} Dataset"
        title = f"Torch Wrapped {base_name}" if not base_name.startswith("Torch") else base_name
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
