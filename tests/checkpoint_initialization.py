import unittest
import torch
from uav.checkpoint_initialization import checkpoint_initialization


class InitializationTests(unittest.TestCase):
    def test_checkpoint_covers_skipped_parameters(self):
        original = torch.nn.Linear(3, 2)
        with checkpoint_initialization():
            model = torch.nn.Linear(3, 2)
            model.load_state_dict(original.state_dict())
        self.assertTrue(torch.equal(original.weight, model.weight))
        self.assertTrue(torch.equal(original.bias, model.bias))

    def test_missing_checkpoint_parameter_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'did not cover'):
            with checkpoint_initialization():
                model = torch.nn.Linear(3, 2)
                model.load_state_dict({'bias': torch.zeros(2)}, strict=False)

    def test_patches_restore_on_error_and_constants_still_initialize(self):
        original = torch.nn.init.uniform_
        with self.assertRaises(ValueError):
            with checkpoint_initialization():
                norm = torch.nn.LayerNorm(3)
                self.assertTrue(torch.equal(norm.weight, torch.ones(3)))
                raise ValueError('construction failed')
        self.assertIs(torch.nn.init.uniform_, original)

    def test_scoped_factory_keeps_other_initialization(self):
        from types import SimpleNamespace
        factory = SimpleNamespace(build=lambda: torch.nn.Linear(3, 2))
        original = factory.build
        reference = torch.nn.Linear(3, 2)
        with checkpoint_initialization(factories=[(factory, 'build')]):
            target = factory.build()
            target.load_state_dict(reference.state_dict())
            # An uncheckpointed module outside the scoped factory is legal.
            other = torch.nn.Linear(3, 2)
            self.assertTrue(torch.isfinite(other.weight).all())
        self.assertIs(factory.build, original)


if __name__ == '__main__':
    unittest.main()
