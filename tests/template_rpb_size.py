"""Configured grids avoid GPU shape scalar reads; no model/media required."""
from types import SimpleNamespace
import unittest
from uav.sam3_resolution import use_configured_rpb_size


class RpbTests(unittest.TestCase):
    def test_grid_switches_and_installation_are_idempotent(self):
        calls=[]
        decoder=SimpleNamespace(compilable_stored_size=(72,72),
            _get_rpb_matrix=lambda boxes,size:calls.append((boxes,size)) or size)
        model=SimpleNamespace(transformer=SimpleNamespace(decoder=decoder))
        use_configured_rpb_size(model);wrapped=decoder._get_rpb_matrix
        use_configured_rpb_size(model)
        self.assertIs(decoder._get_rpb_matrix,wrapped)
        self.assertEqual(wrapped('a',('gpu','scalars')),(72,72))
        decoder.compilable_stored_size=(24,24)
        self.assertEqual(wrapped('b',('gpu','scalars')),(24,24))
        decoder.compilable_stored_size=None
        self.assertEqual(wrapped('c',(48,48)),(48,48))
        self.assertEqual([v[0] for v in calls],['a','b','c'])


if __name__=='__main__':unittest.main()
