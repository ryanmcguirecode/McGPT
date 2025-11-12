from typing import Tuple, Dict
from torch import Tensor


class KVCache:
    """Caches keys and values so we don't have to recompute these at each timestep."""

    _cache: Dict[int, Tuple[Tensor, Tensor]]

    def __init__(self):
        self._cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    def get(self): ...

    def put(self): ...

    def clear(self): ...
