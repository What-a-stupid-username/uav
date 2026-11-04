"""Canonical output selection without modifying original pose evidence."""
import unittest
from tests import template_global_identity as fixtures
from uav.template_mask.output_plan import build_output_plan
from uav.template_mask.geometry import translation
from uav.template_mask.records import Pose


class OutputPlanTests(unittest.TestCase):
    def test_verified_boundary_original_joins_complete_occurrence_without_eight_fragment_samples(self):
        a,_,_=self.owner(0,frames=range(1,24))
        b,obs,_=self.owner(1,frames=range(1))
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        self.e.validated_edge_occurrences=[dict(fragment=b.id,instance=a.id)]
        plan=build_output_plan(self.e)
        self.assertEqual(len(plan.occurrences),1)
        self.assertEqual(set(plan.occurrences[0]),{a.id,b.id})

    setUp=fixtures.GlobalIdentityTests.setUp
    tearDown=fixtures.GlobalIdentityTests.tearDown
    owner=fixtures.GlobalIdentityTests.owner
    def test_duplicate_measurements_have_one_output_owner(self):
        a,_,_=self.owner(0)
        b,_,_=self.owner(1)
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        before={i.id:dict(i.poses) for i in (a,b)}
        plan=build_output_plan(self.e)
        self.assertEqual(len(plan.occurrences),1)
        self.assertTrue(all(len(ids)==1 for ids in plan.frame_owners.values()))
        self.assertEqual(before[a.id],a.poses);self.assertEqual(before[b.id],b.poses)

    def test_shared_template_keeps_remote_occurrences_separate(self):
        self.owner(0)
        self.owner(1,motion=lambda f:translation(300+f,50))
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        plan=build_output_plan(self.e)
        self.assertEqual(len(plan.occurrences),2)
        self.assertTrue(all(len(ids)==2 for ids in plan.frame_owners.values()))

    def test_reappearing_shared_template_does_not_join_lifetimes(self):
        self.owner(0,frames=range(24))
        self.owner(1,frames=range(30,54))
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        plan=build_output_plan(self.e)
        self.assertEqual(len(plan.occurrences),2)

    def test_assigned_original_survives_missing_pose_record(self):
        owner,obs,_=self.owner(0)
        owner.poses.pop(obs.frame)
        obs.ownership_confirmed=True
        plan=build_output_plan(self.e)
        self.assertIn(owner.id,plan.frame_owners[obs.frame])

    def test_assigned_original_survives_when_no_pose_records_remain(self):
        owner,obs,_=self.owner(0)
        owner.poses.clear()
        obs.ownership_confirmed=True
        plan=build_output_plan(self.e)
        self.assertIn(owner.id,plan.frame_owners[obs.frame])

    def test_real_assigned_observation_precedes_duplicate_prediction(self):
        a,_,_=self.owner(0)
        b,obs,_=self.owner(1)
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        frame=obs.frame
        obs.ownership_confirmed=True
        b.poses.pop(frame)
        a.poses[frame]=Pose(frame,translation(),.7,'predicted',reason='missing_match')
        plan=build_output_plan(self.e)
        self.assertEqual(plan.frame_owners[frame],(b.id,))

    def test_invalid_assigned_observation_does_not_create_output_owner(self):
        owner,obs,fine=self.owner(0)
        owner.poses.clear()
        obs.ownership_confirmed=True
        obs.pose=None
        fine.reliable=False
        plan=build_output_plan(self.e)
        self.assertNotIn(owner.id,plan.frame_owners.get(obs.frame,()))

    def test_validated_alias_prediction_survives_old_owner_absence(self):
        a,_,_=self.owner(0)
        b,_,_=self.owner(1)
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        b.alias_of=a.id;b.alias_templates=(0,1)
        a.poses[12]=Pose(12,translation(),1.,'absent',reason='global_detection_absent')
        b.poses[12]=Pose(12,translation(),1.,'predicted',reason='no_original_image_match')
        self.assertFalse(self.e._renders_at(b,12))
        plan=build_output_plan(self.e)
        self.assertEqual(plan.frame_owners[12],(b.id,))
        self.assertEqual(b.poses[12].status,'predicted')

    def test_unresolved_alias_is_not_silenced_by_archived_owner(self):
        a,_,_=self.owner(0)
        b,_,_=self.owner(1,motion=lambda f:translation(300+f,50))
        self.e.bank.propose_link(0,1,translation(),10)
        self.e.bank.propose_link(0,1,translation(),11)
        b.alias_of=a.id;b.alias_templates=(0,1)
        plan=build_output_plan(self.e)
        self.assertEqual(len(plan.occurrences),2)
        self.assertIn(b.id,plan.frame_owners[12])


if __name__=='__main__':unittest.main()
