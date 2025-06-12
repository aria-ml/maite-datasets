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

## Available Datasets

| Dataset  | Task           | Description                                                                    |
|----------|----------------|--------------------------------------------------------------------------------|
| AntiUAV  | Detection      | A UAV detection dataset in natural images with varying backgrounds.            |
| CIFAR10  | Classification | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) classification dataset. |
| MILCO    | Detection      | A side-scan sonar dataset focused on mine-like object detection.               |
| MNIST    | Classification | A dataset of hand-written digits.                                              |
| Seadrone | Detection      | A UAV dataset focused on open water object detection.                          |
| Ships    | Classification | A dataset that focuses on identifying ships from satellite images.             |
| VOC      | Detection      | [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) detection dataset.       |

## Usage

Here is an example of how to import MNIST for usage with your workflow.

```python
>>> from maite_datasets import MNIST

>>> mnist = MNIST(root="data", download=True)
>>> print(mnist)
MNIST Dataset
-------------
    Corruption: None
    Transforms: []
    Image_set: train
    Metadata: {'id': 'MNIST_train', 'index2label': {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}, 'split': 'train'}
    Path: /home/aweng/maite-datasets/data/mnist
    Size: 60000

>>> print("tuple("+", ".join([str(type(t)) for t in mnist[0]])+")")
tuple(<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'dict'>)
```

## Additional Information

Fore more information on the MAITE protocol, check out their [documentation](https://mit-ll-ai-technology.github.io/maite/).

## Acknowledgement

### CDAO Funding Acknowledgement

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
