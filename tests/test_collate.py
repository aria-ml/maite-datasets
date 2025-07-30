import numpy as np
import torch

from maite_datasets._collate import collate_as_list, collate_as_numpy, collate_as_torch


class TestCollateFn:
    def test_list_collate_fn(self):
        assert collate_as_list([("a", 1, 2), ("b", 2, 3), ("c", 3, 4)]) == (["a", "b", "c"], [1, 2, 3], [2, 3, 4])

    def test_list_collate_fn_empty(self):
        assert collate_as_list([]) == ([], [], [])

    def test_numpy_collate_fn(self):
        collated = collate_as_numpy([([1, 2], 1, {"id": 1}), ([3, 4], 2, {"id": 2}), ([5, 6], 3, {"id": 3})])
        assert np.array_equal(collated[0], np.array([[1, 2], [3, 4], [5, 6]]))
        assert collated[1] == [1, 2, 3]
        assert collated[2] == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_numpy_collate_fn_empty(self):
        collated = collate_as_numpy([])
        assert np.array_equal(collated[0], np.array([]))
        assert collated[1] == []
        assert collated[2] == []

    def test_torch_collate_fn(self):
        collated = collate_as_torch([([1, 2], 1, {"id": 1}), ([3, 4], 2, {"id": 2}), ([5, 6], 3, {"id": 3})])
        assert torch.equal(collated[0], torch.tensor([[1, 2], [3, 4], [5, 6]]))
        assert collated[1] == [1, 2, 3]
        assert collated[2] == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_torch_collate_fn_empty(self):
        collated = collate_as_torch([])
        assert torch.equal(collated[0], torch.tensor([]))
        assert collated[1] == []
        assert collated[2] == []
