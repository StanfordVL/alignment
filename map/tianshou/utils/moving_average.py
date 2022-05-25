import torch
import numpy as np
from typing import Union

from tianshou.data import to_numpy


class MovAvg(object):
    """Class for moving average. It will automatically exclude the infinity and
    NaN. Usage:
    ::

        >>> stat = MovAvg(size=66)
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        6.5
        >>> stat.get()
        6.5
        >>> print(f'{stat.mean():.2f}±{stat.std():.2f}')
        6.50±1.12
    """

    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache = []
        self.banned = [np.inf, np.nan, -np.inf]

    def reset(self) -> float:
        self.cache = []
        return self.get()

    def add(self, x: Union[float, list, np.ndarray, torch.Tensor]) -> float:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = to_numpy(x.flatten())
        if isinstance(x, list) or isinstance(x, np.ndarray):
            for _ in x:
                if _ not in self.banned:
                    self.cache.append(_)
        elif x not in self.banned:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)

class VecMovAvg(object):
    """Class for vectorized moving average.
    It will automatically exclude the infinity and NaN.

    Usage:
    ::

        >>> stat = VecMovAvg(ndim=2, size=66)
        >>> stat.add(torch.tensor([5, 2]))
        [5.0, 2.0]
        >>> stat.add(np.array[float('inf'), 3])  # which will not add to stat
        [5.0, 2.5]
        >>> stat.add([6, 7])
        [6.5, 4.0]
        >>> stat.get()
        [6.5, 4.0]
    """

    def __init__(self, size: int = 100, ndim: int = 1) -> None:
        super().__init__()
        self.size = size
        self.ndim = ndim
        self.cache = np.zeros([size, ndim])
        self.length = np.zeros([ndim])
        self.cursor = 0
        self.banned = [np.inf, np.nan, -np.inf]

    def reset(self, start: int, end: int=None) -> np.ndarray:
        if end is None:
            end = start + 1
        self.cache[:, start:end] = 0.0
        self.length[i] = 0.0
        return self.get()

    def add(self, x: Union[list, np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            assert self.ndim == x.size(0)
            x = to_numpy(x)
        if isinstance(x, list) or isinstance(x, np.ndarray):
            assert self.ndim == x.shape[0]
            for i, v in enumerate(x):
                t = np.mean(v)
                if t not in self.banned:
                    self.cache[self.cursor, i] = t
        self.cursor = (self.cursor + 1) % self.size
        return self.get()

    def get(self) -> float:
        """Get the average."""
        return np.sum(self.cache, axis=0) / (self.length + 1e-8)

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache, axis=0)


class Total(object):
    """Class for keeping track of the total.
    It will automatically exclude the infinity and NaN.

    Usage:
    ::

        >>> stat = Total()
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        21
        >>> stat.get()
        21
    """

    def __init__(self) -> None:
        super().__init__()
        self.value = 0
        self.banned = [np.inf, np.nan, -np.inf]

    def reset(self):
        self.value = 0
        return self.get()

    def add(self, x: Union[float, list, np.ndarray, torch.Tensor]) -> float:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = to_numpy(x.flatten())
        if isinstance(x, list) or isinstance(x, np.ndarray):
            for v in x:
                if v not in self.banned:
                    self.value += v
        elif x not in self.banned:
            self.value += x
        return self.get()

    def get(self) -> float:
        """Get the average."""
        return self.value

class VecTotal(object):
    """Class for keeping track of the vector of totals.
    It will automatically exclude the infinity and NaN.

    Usage:
    ::

        >>> stat = VecTotal(ndim=2)
        >>> stat.add(torch.tensor([5, 2]))
        [5.0, 2.0]
        >>> stat.add(np.array[float('inf'), 3)  # which will not add to stat
        [5.0, 5.0]
        >>> stat.add([6, 7])
        [13.0, 14.0]
        >>> stat.get()
        [13.0, 14.0]
    """

    def __init__(self, ndim: int = 1) -> None:
        super().__init__()
        self.ndim = ndim
        self.values = np.zeros(ndim)
        self.banned = [np.inf, np.nan, -np.inf]

    def reset(self, start: int, end: int=None) -> np.ndarray:
        if end is None:
            end = start + 1
        self.values[start:end] = 0.0
        return self.get()

    def add(self, x: Union[list, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            assert self.ndim == x.size(0)
            x = to_numpy(x)
        if isinstance(x, list) or isinstance(x, np.ndarray):
            assert self.ndim == x.shape[0]
            for i, v in enumerate(x):
                t = np.sum(v)
                if t not in self.banned:
                    self.values[i] += t
        return self.get()

    def get(self) -> float:
        """Get the average."""
        return self.values
