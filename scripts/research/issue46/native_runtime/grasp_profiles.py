"""Task-specific historical stance references; actual IK/collisions stay mandatory."""
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np

@lru_cache(maxsize=1)
def load_profiles():
    record=json.loads(Path(__file__).with_name('table_edge_grasp_profiles.json').read_text())
    if record.get('schema')!='robotsim.table-edge-grasp-profiles.v1' or record.get('frame')!='native_scene_world':
        raise ValueError('unsupported grasp profile coordinates')
    return record['profiles']

def select_profile(observed_center, profiles=None):
    center=np.asarray(observed_center,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('finite observed object center required')
    profiles=load_profiles() if profiles is None else profiles
    return min(profiles,key=lambda profile:abs(center[1]-profile['reference_object_y_world_m']))

def stance_from_profile(profile,observed_center,current_root,foot_anchor,current_yaw):
    center=np.asarray(observed_center,dtype=float);root=np.asarray(current_root,dtype=float);anchor=np.asarray(foot_anchor,dtype=float)
    desired=center[:2]+np.asarray(profile['root_offset_from_object_xy_m'])
    yaw=float(profile['base_yaw_rad']);c,s=math.cos(yaw-current_yaw),math.sin(yaw-current_yaw)
    rotation=np.array([[c,-s],[s,c]])
    return desired-rotation@(root-anchor),yaw


def torso_world_heading(profile):
    """Preserve a demonstrated torso heading when the actual base yaw differs."""
    value=float(profile['base_yaw_rad'])+float(profile['waist_yaw_rad'])
    if not math.isfinite(value):raise ValueError('finite profile heading required')
    return math.atan2(math.sin(value),math.cos(value))
