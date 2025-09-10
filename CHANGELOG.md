# Changelog for maite-datasets

## v0.0.8

- [hotfix] Fix `from_huggingface` image classification index2label

## v0.0.7

- [feat] Add HuggingFace dataset adapter support
- [feat] Add support for MAITE datum transforms to base dataset
- [impr] Change ObjectDetectionTarget to `namedtuple` for native Torchvision support
- [deps] Add Python 3.13 support and drop Python 3.9 support

## v0.0.6

- [feat] Add Torchvision convenience wrapper
- [impr] Update base datasets to allow for image or datum transforms

## v0.0.5

- [fix] Fix test and type check errors

## v0.0.4

- [feat] Add COCO and YOLO dataset readers

## v0.0.3

- [feat] Add dataset validation utility
- [feat] Add collate helper functions for dataloaders

## v0.0.2

- [feat] Add dataset builders for custom datasets

## v0.0.1a

- [feat] Update JSON as dictionaries

## v0.0.1

- [feat] Initial release of the `maite-datasets` package
