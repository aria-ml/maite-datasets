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

## Additional Information

For more information on the MAITE protocol, check out their [documentation](https://mit-ll-ai-technology.github.io/maite/).

## Acknowledgement

### CDAO Funding Acknowledgement

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
