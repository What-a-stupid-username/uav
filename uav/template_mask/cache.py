"""Run-local in-memory evidence. No database, serialization or eviction."""
import json
import numpy as np


class EvidenceCache:
    def __init__(self):
        self.memory = {}
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    @staticmethod
    def key(*parts):
        return json.dumps(parts, separators=(',', ':'), sort_keys=True)

    def __contains__(self, key):
        return key in self.memory

    def put(self, key, **arrays):
        arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
        if any(a.dtype.hasobject for a in arrays.values()):
            raise TypeError('object arrays cannot be cached')
        for a in arrays.values():
            a.flags.writeable = False
        previous = self.memory.get(key, {})
        self.bytes += sum(a.nbytes for a in arrays.values())-sum(a.nbytes for a in previous.values())
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.memory[key] = arrays
        return key

    def get(self, key):
        try:
            arrays = self.memory[key]
        except KeyError:
            self.misses += 1
            raise
        self.hits += 1
        return arrays

    def stats(self):
        return dict(storage='memory', entries=len(self.memory), memory_bytes=self.bytes,
                    peak_memory_bytes=self.peak_bytes, hits=self.hits, misses=self.misses,
                    disk_bytes=0)

    def close(self):
        self.memory.clear()
        self.bytes = 0
