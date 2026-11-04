"""One geometry selection rule for final masks and their debug annotations."""
from .pose_filter import output_matrix

_UNSELECTED = object()

def selected_observation(engine, instance, frame, observations=None):
    if observations is None:
        observations = engine.associated_observations(instance, frame)
    return min(observations, key=lambda o: (
        (o.trajectory_evidence or {}).get('coordinate_registration') == 'predicted',
        not o.reliable)) if observations else None


def selected_matrix(engine, instance, frame, observation=_UNSELECTED):
    if observation is _UNSELECTED:
        observation = selected_observation(engine, instance, frame)
    pose = instance.poses.get(frame)
    if observation is None and (pose is None or pose.status=='absent'):
        return None
    matrix = observation.pose if observation is not None else pose.matrix if pose is not None else None
    if matrix is None:
        return None
    return output_matrix(instance, frame, matrix,
                         engine.bank.templates[instance.template_id].anchor)
