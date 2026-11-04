"""Live identity sampling retains independent sources and full later evidence."""
import unittest
from types import SimpleNamespace
from uav.template_mask.reconciliation import TemplateReconciler
from uav.template_mask.records import Reference

class IdentityQueryTests(unittest.TestCase):
    def test_temporal_groups_select_quality_and_keep_original_order(self):
        refs=[Reference(str(i),i,'features',q) for i,q in enumerate([1,9,2,3,8,4])]
        observations={r.observation_id:SimpleNamespace(reliable=True,ambiguous=False) for r in refs}
        engine=SimpleNamespace(stage='first_pass',bank=SimpleNamespace(observations=observations))
        template=SimpleNamespace(references=refs)
        reconciler=TemplateReconciler(engine)
        self.assertEqual([r.source_id for r in reconciler.query_references(template)],[1,4])
        self.assertIs(template.references,refs)
        observations['1'].ambiguous=True
        self.assertNotIn(refs[1],reconciler.query_references(template))
        engine.stage='second_pass'
        self.assertIs(reconciler.query_references(template),refs)

    def test_sparse_views_are_not_duplicated(self):
        refs=[Reference('a',0,'features',1.)]
        engine=SimpleNamespace(stage='first_pass',bank=SimpleNamespace(
            observations={'a':SimpleNamespace(reliable=True,ambiguous=False)}))
        self.assertEqual(TemplateReconciler(engine).query_references(SimpleNamespace(references=refs)),refs)
