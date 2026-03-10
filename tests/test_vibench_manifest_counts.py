from src.agents.deep_model_train_agent import _loader_batch_count, _loader_dataset_count


class _DummyBaseLoader:
    def __init__(self, dataset_size: int, batch_count: int):
        self.dataset = list(range(dataset_size))
        self._batch_count = int(batch_count)

    def __len__(self):
        return self._batch_count


class _WrappedLoader:
    def __init__(self, base):
        self._base = base

    def __len__(self):
        return len(self._base)


def test_loader_count_helpers_use_underlying_dataset_and_batches():
    base = _DummyBaseLoader(dataset_size=23, batch_count=4)
    wrapped = _WrappedLoader(base)
    assert _loader_dataset_count(wrapped) == 23
    assert _loader_batch_count(wrapped) == 4


def test_loader_dataset_count_gracefully_handles_missing_dataset():
    class _NoDataset:
        def __len__(self):
            return 3

    loader = _NoDataset()
    assert _loader_dataset_count(loader) == 0
    assert _loader_batch_count(loader) == 3
