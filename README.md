# MAITE Datasets

MAITE Datasets are a collection of public datasets wrapped in a [MAITE](https://mit-ll-ai-technology.github.io/maite/) compliant format.

## Installation

To install and use `maite-datasets` you can use pip:

```bash
pip install maite-datasets
```

For status bar indicators when downloading, you can include the extra `tqdm` when installing:

```bash
pip install maite-datasets[tqdm]
```

## Available Downloadable Datasets

| Task           | Dataset          | Description                                                         |
|----------------|------------------|---------------------------------------------------------------------|
| Classification | CIFAR10          | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.     |
| Classification | MNIST            | A dataset of hand-written digits.                                   |
| Classification | Ships            | A dataset that focuses on identifying ships from satellite images.  |
| Detection      | AntiUAVDetection | A UAV detection dataset in natural images with varying backgrounds. |
| Detection      | MILCO            | A side-scan sonar dataset focused on mine-like object detection.    |
| Detection      | Seadrone         | A UAV dataset focused on open water object detection.               |
| Detection      | VOCDetection     | [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.      |

### Usage

Here is an example of how to import MNIST for usage with your workflow.

```python
>>> from maite_datasets.image_classification import MNIST

>>> mnist = MNIST(root="data", download=True)
>>> print(mnist)
MNIST Dataset
-------------
    Corruption: None
    Transforms: []
    Image_set: train
    Metadata: {'id': 'MNIST_train', 'index2label': {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}, 'split': 'train'}
    Path: /home/user/maite-datasets/data/mnist
    Size: 60000

>>> print("tuple("+", ".join([str(type(t)) for t in mnist[0]])+")")
tuple(<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'dict'>)
```

## Dataset Wrappers

Wrappers provide a way to convert datasets to allow usage of tools within specific backend frameworks.

### Torchvision

`TorchvisionWrapper` is a convenience class that wraps any of the datasets and provides the capability to apply
`torchvision` transforms to the dataset.

**NOTE:** `TorchvisionWrapper` requires _torch_ and _torchvision_ to be installed.

```python
>>> from maite_datasets.object_detection import MILCO

>>> milco = MILCO(root="data", download=True)
>>> print(milco)
MILCO Dataset
-------------
    Transforms: []
    Image Set: train
    Metadata: {'id': 'MILCO_train', 'index2label': {0: 'MILCO', 1: 'NOMBO'}, 'split': 'train'}
    Path: /home/user/maite-datasets/data/milco
    Size: 261

>>> print(f"type={milco[0][0].__class__.__name__}, shape={milco[0][0].shape}")
type=ndarray, shape=(3, 1024, 1024)

>>> print(milco[0][1].boxes[0])
[ 75. 217. 130. 247.]

>>> from maite_datasets.wrappers import TorchvisionWrapper
>>> from torchvision.transforms.v2 import Resize

>>> milco_torch = TorchvisionWrapper(milco, transforms=Resize(224))
>>> print(milco_torch)
Torchvision Wrapped MILCO Dataset
---------------------------
    Transforms: Resize(size=[224], interpolation=InterpolationMode.BILINEAR, antialias=True)

MILCO Dataset
-------------
    Transforms: []
    Image Set: train
    Metadata: {'id': 'MILCO_train', 'index2label': {0: 'MILCO', 1: 'NOMBO'}, 'split': 'train'}
    Path: /home/user/maite-datasets/data/milco
    Size: 261

>>> print(f"type={milco_torch[0][0].__class__.__name__}, shape={milco_torch[0][0].shape}")
type=Image, shape=torch.Size([3, 224, 224])

>>> print(milco_torch[0][1].boxes[0])
tensor([16.4062, 47.4688, 28.4375, 54.0312], dtype=torch.float64)
```

## Dataset Adapters

Adapters provide a way to read in datasets from other popular formats.

### Huggingface

Hugging face datasets can be adapted into MAITE compliant format using the `from_huggingface` adapter.

```python
>>> from datasets import load_dataset
>>> from maite_datasets.adapters import from_huggingface

>>> cppe5 = load_dataset("cppe-5")
>>> m_cppe5 = from_huggingface(cppe5["train"])
>>> print(m_cppe5)
HFObjectDetection Dataset
-------------------------
    Source: Dataset({
    features: ['image_id', 'image', 'width', 'height', 'objects'],
    num_rows: 1000
})
    Metadata: {'id': 'cppe-5', 'index2label': {0: 'Coverall', 1: 'Face_Shield', 2: 'Gloves', 3: 'Goggles', 4: 'Mask'}, 'description': '', 'citation': '', 'homepage': '', 'license': '', 'features': {'image_id': Value('int64'), 'image': Image(mode=None, decode=True), 'width': Value('int32'), 'height': Value('int32'), 'objects': {'id': List(Value('int64')), 'area': List(Value('int64')), 'bbox': List(List(Value('float32'), length=4)), 'category': List(ClassLabel(names=['Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask']))}}, 'post_processed': None, 'supervised_keys': None, 'builder_name': 'parquet', 'dataset_name': 'cppe-5', 'config_name': 'default', 'version': 0.0.0, 'splits': {'train': SplitInfo(name='train', num_bytes=240478590, num_examples=1000, shard_lengths=None, dataset_name='cppe-5'), 'test': SplitInfo(name='test', num_bytes=4172706, num_examples=29, shard_lengths=None, dataset_name='cppe-5')}, 'download_checksums': {'hf://datasets/cppe-5@66f6a5efd474e35bd7cb94bf15dea27d4c6ad3f8/data/train-00000-of-00001.parquet': {'num_bytes': 237015519, 'checksum': None}, 'hf://datasets/cppe-5@66f6a5efd474e35bd7cb94bf15dea27d4c6ad3f8/data/test-00000-of-00001.parquet': {'num_bytes': 4137134, 'checksum': None}}, 'download_size': 241152653, 'post_processing_size': None, 'dataset_size': 244651296, 'size_in_bytes': 485803949}

>>> image = m_cppe5[0][0]
>>> print(f"type={image.__class__.__name__}, shape={image.shape}")
type=ndarray, shape=(3, 663, 943)

>>> target = m_cppe5[0][1]
>>> print(f"box={target.boxes[0]}, label={target.labels[0]}")
box=[302.0, 109.0, 73.0, 52.0], label=4

>>> print(m_cppe5[0][2])
{'id': [114, 115, 116, 117], 'image_id': 15, 'width': 943, 'height': 663, 'area': [3796, 1596, 152768, 81002]}
```

## Additional Information

For more information on the MAITE protocol, check out their [documentation](https://mit-ll-ai-technology.github.io/maite/).

## Acknowledgement

### CDAO Funding Acknowledgement

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
