"""Calibrated SIMULATION_ONLY head RGB-D, using the existing renderer/FK bridge.

The camera is fixed to the actual head; it never looks at an object truth pose.
Only sensor generation sees the simulated scene. Cylinder fitting consumes
RGB, metric depth, intrinsics and camera extrinsics, without body IDs or masks.
This is an explicitly calibrated virtual sensor, not a verified vendor optical
extrinsic or real-device depth/noise model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
from scripts.agent import path_governance as pg

CAMERA='issue46_calibrated_head_rgbd'
WIDTH,HEIGHT,FOVY=320,240,60.

def render_backend_info():
    from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
    values={name:glGetString(key) for name,key in (('vendor',GL_VENDOR),('renderer',GL_RENDERER),('version',GL_VERSION))}
    if any(value is None for value in values.values()):raise RuntimeError('RGBD_GL_CONTEXT_IDENTITY_UNAVAILABLE')
    return {name:value.decode() for name,value in values.items()}

def compile_sensor_model(scene):
    from scripts.research.issue46.perception.rgbd_truth_capture import CAMERA_POS
    root=ET.fromstring(Path(scene).read_bytes())
    parent=root.find('.//body[@name="head_pitch_link"]')
    if parent is None:raise ValueError('actual head body missing')
    # Fixed head-relative virtual calibration: 0.65rad downward, +X head
    # forward, camera +X right/+Y up/-Z forward. No object look-at call.
    a=.65;R=np.array([[0.,math.sin(a),-math.cos(a)],[-1.,0.,0.],[0.,math.cos(a),math.sin(a)]])
    ET.SubElement(parent,'camera',name=CAMERA,pos=CAMERA_POS,xyaxes=' '.join(map(str,np.r_[R[:,0],R[:,1]])),fovy=str(FOVY))
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))

def prepare_host_cache(scene):
    """Compile before native physics; adding a camera has no physical effect."""
    scene=Path(scene);source_hash=hashlib.sha256(scene.read_bytes()).hexdigest();code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parent=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
    for existing in sorted(parent.glob('native-rgbd-host-cache-*/result.json'),reverse=True):
        record=json.loads(existing.read_text())
        if record.get('source_sha256')==source_hash and record.get('sensor_source_sha256')==code_hash and record.get('engine')==mujoco.mj_version():
            load_cached_model(scene,existing);return existing
    out=pg.create_unique_output_directory(parent,prefix='native-rgbd-host-cache').native_path
    original=mujoco.MjModel.from_xml_path(str(scene));model=compile_sensor_model(scene)
    checked=[]
    for key in ('body_parentid','body_pos','body_quat','body_mass','body_inertia','body_gravcomp','jnt_type','jnt_range','jnt_pos','jnt_axis','dof_frictionloss','dof_damping','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','eq_type','eq_obj1id','eq_obj2id','eq_data','actuator_trnid','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','qpos0','key_qpos'):
        if not np.array_equal(getattr(original,key),getattr(model,key),equal_nan=True):raise RuntimeError('sensor cache changed physical array:'+key)
        checked.append(key)
    binary=out/'host_sensor_and_ik.mjb';mujoco.mj_saveModel(model,str(binary),None)
    record={'classification':'OFFLINE_HOST_COMPILED_CACHE_CAMERA_ONLY_NATIVE_XML_UNCHANGED','source':str(scene),'source_sha256':source_hash,'sensor_source_sha256':code_hash,'engine':mujoco.mj_version(),'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'unchanged_physical_arrays':checked,'native_engine_uses_original_xml':True,'mj_step_calls':0}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);return out/'result.json'

def load_cached_model(scene,cache):
    record=json.loads(Path(cache).read_text());binary=Path(record['binary'])
    if record.get('source_sha256')!=hashlib.sha256(Path(scene).read_bytes()).hexdigest() or record.get('engine')!=mujoco.mj_version() or record.get('sensor_source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or record.get('binary_sha256')!=hashlib.sha256(binary.read_bytes()).hexdigest():raise ValueError('host compiled model cache hash/version mismatch')
    return mujoco.MjModel.from_binary_path(str(binary))

def build_sensor(scene,cache=None):
    model=load_cached_model(scene,cache) if cache else compile_sensor_model(scene)
    camera=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,CAMERA)
    return model,mujoco.MjData(model),mujoco.Renderer(model,height=HEIGHT,width=WIDTH),camera

def render(model,data,renderer,camera,row):
    # Rendering requires FK/camera transforms, not another collision/constraint
    # solve in this observation process. Native physics remains untouched.
    apply_recorded_native_frame(model,data,row,visual_kinematics_only=True)
    if row['bodies'].get('head_pitch_link'):
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'head_pitch_link')
        measured=row['bodies']['head_pitch_link']
        if np.linalg.norm(data.xpos[bid]-measured['position_m'])>.001 or np.max(np.abs(data.xmat[bid]-measured['rotation']))>.001:raise RuntimeError('CAMERA_HEAD_NATIVE_FK_MISMATCH')
    def update():
        renderer.update_scene(data,camera=CAMERA)
        # The virtual sensor needs the direct camera view, not the expensive
        # aesthetic mirror/shadow passes of the human-review window.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
    update();rgb=renderer.render()
    renderer.enable_depth_rendering();update();depth=renderer.render();renderer.disable_depth_rendering()
    T=np.eye(4);T[:3,:3]=data.cam_xmat[camera].reshape(3,3)@np.diag([1.,-1.,-1.]);T[:3,3]=data.cam_xpos[camera]
    f=HEIGHT/(2*math.tan(math.radians(FOVY)/2))
    return rgb,depth,{'fx':f,'fy':f,'cx':(WIDTH-1)/2,'cy':(HEIGHT-1)/2},T

def vertical_component(mask, *, world_height=None):
    """Select a 4-connected blue component with real upright height support.

    A tilted camera can make an upright bottle appear squat or diagonal.
    Calibrated depth therefore qualifies world height; the image-only fallback
    retains the old aspect rule for callers without calibrated geometry.
    """

    from scipy.ndimage import label,find_objects,binary_erosion
    labels,count=label(mask);sizes=np.bincount(labels.reshape(-1));best=None
    for index,bounds in enumerate(find_objects(labels),1):
        if bounds is None or sizes[index]<50:continue
        ys,xs=bounds;height=ys.stop-ys.start;width=xs.stop-xs.start;aspect=height/max(width,1)
        if world_height is None:
            if not 1.25<=aspect<=8.:continue
            score=float(sizes[index])*min(aspect/3.5,3.5/aspect)
        else:
            values=world_height[labels==index]
            if not np.isfinite(values).all():continue
            # Qualify the same interior depth that the cylinder fit consumes.
            # RGB antialiasing along floor/foreground boundaries can carry a
            # different surface depth and falsely give a flat floor height.
            interior=binary_erosion(labels==index)
            values=world_height[interior]
            if len(values)<20:continue
            lo,hi=np.quantile(values,[.02,.98]);span=float(hi-lo)
            if span<.06:continue
            # Apply the fitter's existing side-depth support before choosing
            # the largest component. Floor+wall fragments can have a tall
            # height range but no points along a continuous cylinder barrel.
            side_count=int(np.count_nonzero((values>lo+.15*span)&(values<hi-.20*span)))
            if side_count<60:continue
            score=float(len(values))
        info={'pixel_count':float(sizes[index]),'bbox_xyxy':[xs.start,ys.start,xs.stop,ys.stop],'bbox_width_px':float(width),'bbox_height_px':float(height),'vertical_aspect':float(aspect),'score':score}
        if best is None or score>best[1]['score']:best=(labels==index,info)
    return best

def occluded_cap_points(color_mask, barrel_mask, depth, K, T, circle, barrel_upper):
    """Associate a disconnected visible cap using measured 3D geometry only.

    A foreground arm can split one cylinder into two image components. The
    additional component must lie inside the fitted barrel, above it within
    one diameter, and contain a spatially supported horizontal top plane.
    Multiple qualifying caps are ambiguous and are rejected.
    """
    from scipy.ndimage import binary_erosion, label, find_objects
    labels, _ = label(color_mask & ~barrel_mask)
    candidates=[]
    for index, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:continue
        interior=binary_erosion(labels==index);y,x=np.where(interior)
        if len(x)<20:continue
        z=depth[y,x]
        optical=np.c_[(x-K['cx'])*z/K['fx'],(y-K['cy'])*z/K['fy'],z]
        points=optical@T[:3,:3].T+T[:3,3]
        lo,hi=np.quantile(points[:,2],[.02,.98])
        if lo<barrel_upper-.003 or lo-barrel_upper>2*circle[2]:continue
        radial=np.linalg.norm(points[:,:2]-circle[:2],axis=1)
        if np.mean(radial<=circle[2]+.003)<.95:continue
        top=points[points[:,2]>=hi-.0015]
        if len(top)<20 or np.std(top[:,2])>.003:continue
        if np.linalg.eigvalsh(np.cov(top[:,:2].T))[0]<.002**2:continue
        candidates.append((top,{'component_id':index,'depth_pixels':len(points),
                               'top_pixels':len(top),'occluded_axial_gap_m':float(lo-barrel_upper),
                               'inside_barrel_fraction':float(np.mean(radial<=circle[2]+.003))}))
    if len(candidates)>1:raise ValueError('AMBIGUOUS_DISCONNECTED_CYLINDER_CAP')
    return candidates[0] if candidates else (None,None)

def supported_cylinder_top(depth, K, T, circle, visible_bottom, known_height_m):
    """Infer a supported upright object's top from a measured local table plane.

    Allowed only for the pre-contact task that declares object height and table
    support. It is not valid for an airborne bottle. No scene/object pose input.
    """
    if not .05<=known_height_m<=.50:raise ValueError('KNOWN_OBJECT_HEIGHT_OUT_OF_TASK_RANGE')
    y,x=np.indices(depth.shape);valid=np.isfinite(depth)&(depth>.08)&(depth<1.2)
    z=np.where(valid,depth,0.)
    points=np.stack(((x-K['cx'])*z/K['fx'],(y-K['cy'])*z/K['fy'],z),axis=-1)@T[:3,:3].T+T[:3,3]
    radial=np.linalg.norm(points[:,:,:2]-circle[:2],axis=-1)
    low=visible_bottom-known_height_m
    region=valid&(radial>circle[2]+.01)&(radial<circle[2]+.15)&(points[:,:,2]<=visible_bottom+.003)&(points[:,:,2]>=low)
    candidates=points[region]
    if len(candidates)<30:raise ValueError('INSUFFICIENT_LOCAL_SUPPORT_PLANE')
    counts,bins=np.histogram(candidates[:,2],bins=np.arange(low,visible_bottom+.005,.001))
    center=bins[int(np.argmax(counts))]+.0005
    plane=candidates[np.abs(candidates[:,2]-center)<=.0015]
    if len(plane)<30 or np.std(plane[:,2])>.001:raise ValueError('LOCAL_SUPPORT_PLANE_RESIDUAL')
    if np.linalg.eigvalsh(np.cov(plane[:,:2].T))[0]<.01**2:raise ValueError('LOCAL_SUPPORT_PLANE_SPATIAL_SUPPORT')
    bottom=float(np.median(plane[:,2]));top=bottom+known_height_m
    return top,{'source':'RGBD_LOCAL_TABLE_PLANE_PLUS_DECLARED_HEIGHT',
                'known_object_height_m':known_height_m,'support_plane_z_m':bottom,
                'support_plane_pixels':len(plane),'support_plane_std_m':float(np.std(plane[:,2])),
                'validity':'UPRIGHT_TABLE_SUPPORTED_PRECONTACT_ONLY'}


def estimate(rgb,depth,K,T, *, supported_height_m=None):
    """Task-blue, upright variable-radius cylinder. No ground-truth input."""
    red,green,blue=(rgb[...,i].astype(np.int16) for i in range(3))
    mask=(blue>=70)&(blue-red>=28)&(blue-green>=18)&np.isfinite(depth)&(depth>.08)&(depth<1.2)
    ys,xs=np.indices(depth.shape)
    proposal_depth=np.where(mask,depth,0.)
    world_height=(T[2,0]*(xs-K["cx"])*proposal_depth/K["fx"]+
                  T[2,1]*(ys-K["cy"])*proposal_depth/K["fy"]+T[2,2]*proposal_depth+T[2,3])
    color_mask=mask.copy()
    component=vertical_component(mask,world_height=world_height)
    if component is None:raise ValueError('NO_BLUE_CYLINDER_RGBD_COMPONENT')
    mask,info=component
    # RGB antialiasing and a foreground finger can mix colour/depth at a
    # silhouette. Exclude one image-boundary pixel; this does not shrink any
    # physical collision surface or substitute a truth segmentation mask.
    from scipy.ndimage import binary_erosion
    interior=binary_erosion(mask);y,x=np.where(interior);z=depth[y,x]
    optical=np.c_[(x-K['cx'])*z/K['fx'],(y-K['cy'])*z/K['fy'],z]
    points=optical@T[:3,:3].T+T[:3,3]
    lo,hi=np.quantile(points[:,2],[.02,.98])
    if hi-lo<.06:raise ValueError('INSUFFICIENT_VISIBLE_CYLINDER_HEIGHT')
    side=points[(points[:,2]>lo+.15*(hi-lo))&(points[:,2]<hi-.20*(hi-lo)),:2]
    if len(side)<60:raise ValueError('INSUFFICIENT_SIDE_DEPTH')
    if len(side)>2500:side=side[::math.ceil(len(side)/2500)]
    A=np.c_[2*side,np.ones(len(side))];v=np.linalg.lstsq(A,np.sum(side*side,axis=1),rcond=None)[0]
    radius=math.sqrt(max(.005**2,float(v[2]+v[0]**2+v[1]**2)))
    seed=np.r_[v[:2],np.clip(radius,.006,.079)]
    seeds=[seed]
    # A few foreground depth outliers make the algebraic circle ill-conditioned.
    # Bounded deterministic three-point hypotheses use only this frame's depth.
    rng=np.random.default_rng(46)
    for _ in range(32):
        triple=side[rng.choice(len(side),3,replace=False)];matrix=2*(triple[1:]-triple[0])
        if abs(np.linalg.det(matrix))<1e-7:continue
        c=np.linalg.solve(matrix,np.sum(triple[1:]**2,axis=1)-np.sum(triple[0]**2));r=float(np.linalg.norm(c-triple[0]))
        if .005<r<.08:seeds.append(np.r_[c,r])
    seed=min(seeds,key=lambda q:float(np.median(np.abs(np.linalg.norm(side-q[:2],axis=1)-q[2]))))
    fit=least_squares(lambda q:np.linalg.norm(side-q[:2],axis=1)-q[2],seed,bounds=([side[:,0].min()-.1,side[:,1].min()-.1,.005],[side[:,0].max()+.1,side[:,1].max()+.1,.08]),loss='soft_l1',f_scale=.001,max_nfev=80)
    inliers=np.abs(np.linalg.norm(side-fit.x[:2],axis=1)-fit.x[2])<=.002
    if float(np.mean(inliers))<.80:raise ValueError('CYLINDER_SIDE_INLIER_SUPPORT_UNQUALIFIED')
    fit=least_squares(lambda q:np.linalg.norm(side[inliers]-q[:2],axis=1)-q[2],fit.x,bounds=([side[:,0].min()-.1,side[:,1].min()-.1,.005],[side[:,0].max()+.1,side[:,1].max()+.1,.08]),loss='soft_l1',f_scale=.001,max_nfev=40)
    residual=np.abs(np.linalg.norm(side-fit.x[:2],axis=1)-fit.x[2]);angles=np.sort(np.mod(np.arctan2(side[inliers,1]-fit.x[1],side[inliers,0]-fit.x[0]),2*math.pi))
    # Largest empty angular gap removes the wrap discontinuity.
    angle_support=2*math.pi-float(np.max(np.diff(np.r_[angles,angles[0]+2*math.pi])))
    # A narrow bottle exposes fewer than 10% top pixels. A fixed quantile
    # fraction mixes its barrel into the cap. Require an actual horizontal
    # top band near the robust upper extent instead, with unchanged residual
    # and noise checks.
    top=points[points[:,2]>=hi-.0015,2]
    cap,cap_association=occluded_cap_points(color_mask,mask,depth,K,T,fit.x,float(hi))
    if cap is not None:top=cap[:,2]
    height_evidence={'source':'DIRECT_RGBD_TOP_PLANE'}
    if supported_height_m is not None:
        top_z,height_evidence=supported_cylinder_top(depth,K,T,fit.x,float(lo),supported_height_m)
        if float(hi)>top_z+.005:raise ValueError('VISIBLE_CYLINDER_ABOVE_DECLARED_SUPPORTED_HEIGHT')
    else:
        if len(top)<20:raise ValueError('INSUFFICIENT_TOP_PLANE_SUPPORT')
        top_z=float(np.median(top))
    if np.median(residual)>.002 or angle_support<.6 or (supported_height_m is None and np.std(top)>.003):raise ValueError('CYLINDER_FIT_SUPPORT_UNQUALIFIED:'+json.dumps({'residual_m':float(np.median(residual)),'angular_support_rad':angle_support,'top_std_m':float(np.std(top)),'fit':fit.x.tolist(),'height_range':[float(lo),float(hi)],'component':info}))
    return {'center_xy_world_m':fit.x[:2].tolist(),'radius_m':float(fit.x[2]),'diameter_m':float(2*fit.x[2]),'top_world_z_m':top_z,'visible_bottom_world_z_m':float(lo),'visible_height_m':float(top_z-lo),'axis_world':[0.,0.,1.],'axis_assumption':'UPRIGHT_GRAVITY_ALIGNED_TASK_OBJECT','median_surface_residual_m':float(np.median(residual)),'side_inlier_fraction':float(np.mean(inliers)),'angular_support_rad':angle_support,'depth_pixels':len(points),'excluded_boundary_pixels':int(mask.sum()-interior.sum()),'rgb_component':info,'disconnected_cap_association':cap_association,'height_evidence':height_evidence},mask

def capture_once(scene,row,output):
    from PIL import Image
    model,data,renderer,camera=build_sensor(scene)
    try:rgb,depth,K,T=render(model,data,renderer,camera,row)
    finally:renderer.close()
    Image.fromarray(rgb).save(output/'rgb.png');np.save(output/'depth_m.npy',depth)
    result,mask=estimate(rgb,depth,K,T)
    Image.fromarray(np.uint8(mask)*255).save(output/'rgb_mask.png')
    result.update({'classification':'SIMULATION_ONLY_CALIBRATED_HEAD_RGBD_VARIABLE_RADIUS','source_host_ns':row['host_monotonic_ns'],'generation':row['generation'],'camera_intrinsics':K,'T_world_optical':T.tolist(),'object_truth_used_by_estimator':False,'geometry_or_head_joint_changed':False,'vendor_optical_calibration_verified':False})
    return result

def qualify_estimate(record, *, generation, now_ns=None):
    """Control-side contract. Never accepts missing, old or evaluation truth."""
    now=time.monotonic_ns() if now_ns is None else now_ns
    if record.get('pass') is not True or record.get('generation')!=generation:raise RuntimeError('RGBD_NOT_QUALIFIED_OR_WRONG_GENERATION:'+str(record.get('failure','generation mismatch')))
    age=(now-int(record['source_host_ns']))*1e-9
    if not 0.<=age<=.20:raise RuntimeError('STALE_RGBD_SOURCE:'+str(age))
    xyz=np.asarray(record['estimated_center_world_m'],dtype=float);axis=np.asarray(record['axis_world'],dtype=float)
    if xyz.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[xyz,axis,record['diameter_m']]).all() or not np.allclose(axis,[0.,0.,1.],atol=1e-8):raise RuntimeError('INVALID_RGBD_POSE')
    if not .01<=record['diameter_m']<=.16 or record.get('object_truth_used_by_estimator') is not False:raise RuntimeError('INVALID_RGBD_PROVENANCE_OR_SIZE')
    return {'position_m':xyz.tolist(),'rotation':np.eye(3).reshape(-1).tolist(),'quaternion_wxyz':[1.,0.,0.,0.]}

def await_initial_estimate(path, *, generation, timeout_s=2.,after_source_ns=None):
    """No-motion readiness only. Active servo calls qualify_estimate directly."""
    if not 0.<timeout_s<=2.:raise ValueError('initial image readiness budget exceeds 2s')
    end=time.monotonic()+timeout_s
    while True:
        record=json.loads(Path(path).read_text())
        if after_source_ns is not None and record.get('pass') is True and int(record['source_host_ns'])<=after_source_ns:
            if time.monotonic()>=end:raise RuntimeError('NO_ADVANCING_RGBD_FOR_INITIAL_READINESS')
            time.sleep(.01);continue
        try:return record,qualify_estimate(record,generation=generation)
        except RuntimeError as exc:
            if not str(exc).startswith('STALE_RGBD_SOURCE') or time.monotonic()>=end:raise
            time.sleep(.01)

def make_stationary_snapshot(planned,verification, *, generation,now_ns=None):
    now=time.monotonic_ns() if now_ns is None else now_ns
    qualify_estimate(planned,generation=generation,now_ns=planned['created_host_ns'])
    qualify_estimate(verification,generation=generation,now_ns=now)
    if verification['source_host_ns']<=planned['source_host_ns']:raise RuntimeError('SNAPSHOT_REQUIRES_INDEPENDENT_FRESH_REOBSERVATION')
    error=float(np.linalg.norm(np.asarray(planned['estimated_center_world_m'])-verification['estimated_center_world_m']))
    if error>.002 or abs(planned['diameter_m']-verification['diameter_m'])>.002:raise RuntimeError('OBJECT_NOT_STATIONARY_BETWEEN_RGBD_OBSERVATIONS')
    return {'schema':'robotsim.issue46.stationary-rgbd-grasp-snapshot.v1','classification':'FRESH_ACQUISITION_BOUNDED_STATIC_WORLD_TARGET_NOT_CONTINUOUS_VISION','generation':generation,'frozen_at_host_ns':now,'valid_for_s':45.,'estimated_center_world_m':list(planned['estimated_center_world_m']),'axis_world':list(planned['axis_world']),'diameter_m':planned['diameter_m'],'original_observation':json.loads(json.dumps(planned)),'pre_execution_verification':json.loads(json.dumps(verification)),'reobservation_position_difference_m':error,'expected_object_stationary':True,'object_truth_used_by_estimator':False,'independent_native_motion_contact_and_fall_guards_unchanged':True}

def qualify_stationary_snapshot(record, *, generation,now_ns=None):
    now=time.monotonic_ns() if now_ns is None else now_ns
    if record.get('schema')!='robotsim.issue46.stationary-rgbd-grasp-snapshot.v1' or record.get('generation')!=generation or record.get('valid_for_s')!=45. or record.get('expected_object_stationary') is not True:raise RuntimeError('INVALID_RGBD_SNAPSHOT_CONTRACT')
    frozen=int(record['frozen_at_host_ns'])
    if not 0.<=(now-frozen)*1e-9<=45.:raise RuntimeError('RGBD_SNAPSHOT_EXECUTION_DEADLINE')
    rebuilt=make_stationary_snapshot(record['original_observation'],record['pre_execution_verification'],generation=generation,now_ns=frozen)
    if any(record[k]!=rebuilt[k] for k in ('estimated_center_world_m','axis_world','diameter_m','object_truth_used_by_estimator')):raise RuntimeError('RGBD_SNAPSHOT_TARGET_CHANGED')
    return {'position_m':record['estimated_center_world_m'],'rotation':np.eye(3).reshape(-1).tolist(),'quaternion_wxyz':[1.,0.,0.,0.]}

def stream(scene,latest,output, *, half_height_m,cache=None):
    from PIL import Image
    from scripts.research.issue46.perception.rgbd_truth_capture import CAMERA_POS
    if half_height_m not in (.075,.11,.15):raise ValueError('explicit canonical height prior required')
    output.mkdir(exist_ok=False);model,data,renderer,camera=build_sensor(scene,cache)
    backend=render_backend_info();requested=os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME')
    if requested and requested.lower() not in backend['renderer'].lower():raise RuntimeError('RGBD_RENDER_ADAPTER_SELECTION_NOT_CONFIRMED')
    configuration={'schema':'robotsim.issue46.calibrated-virtual-rgbd.v1','scene':str(scene),'scene_sha256':hashlib.sha256(Path(scene).read_bytes()).hexdigest(),'sensor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'camera_parent':'head_pitch_link','position_parent_m':CAMERA_POS,'fixed_downward_angle_rad':.65,'fovy_deg':FOVY,'resolution':[WIDTH,HEIGHT],'sensor_render_shadows':False,'sensor_render_reflections':False,'renderer_engine':mujoco.mj_version(),'no_object_lookat':True,'vendor_optical_calibration_verified':False,'height_prior_m':2*half_height_m,'size_estimator':'variable-radius visible cylinder with horizontal top plane','robot_physics_changed':False}
    configuration['graphics_backend']=backend;configuration['requested_render_adapter']=requested
    pg.write_new_text(output/'sensor_configuration.json',json.dumps(configuration,indent=2)+'\n')
    # EGL/shader first-use initialization is not a qualified observation.
    # Warm it before acquiring the source stamp used by active control.
    warm_start=time.monotonic_ns();warm_row=json.loads(latest.read_text())
    render(model,data,renderer,camera,warm_row)
    pg.write_new_text(output/'warmup.json',json.dumps({'classification':'RENDER_INITIALIZATION_NOT_CONTROL_FEEDBACK','elapsed_s':(time.monotonic_ns()-warm_start)*1e-9,'source_stamp_not_published':warm_row['host_monotonic_ns']})+'\n')
    trace=[];dropped=[];last_unique=None;deadline=time.monotonic()+90.;generation=None
    def publish(record):
        temporary=output/'latest.tmp';temporary.write_text(json.dumps(record)+'\n');os.replace(temporary,output/'latest.json')
    try:
        while time.monotonic()<deadline and not (output/'stop').exists():
            phase_path=output.parent/'native_tracking_phase.json'
            if phase_path.exists():
                phase=json.loads(phase_path.read_text())
                if phase.get('phase') in ('ARRIVAL_HOLD','CLOSING','LIFTING') and phase.get('native_arrival_pass') is True:
                    pg.write_new_text(output/'contact_handoff.json',json.dumps({'classification':'PERCEPTION_TO_FIXED_TARGET_AND_PROPRIOCEPTIVE_CONTACT_HANDOFF','phase':phase,'last_sensor_source_ns':last_unique,'freshness_gates_before_closure_unchanged':True})+'\n');break
            tick=time.monotonic();row=json.loads(latest.read_text());stamp=row['host_monotonic_ns']
            if not 0.<=(time.monotonic_ns()-stamp)*1e-9<=.20:raise RuntimeError('STALE_SENSOR_GENERATION_INPUT')
            if generation is not None and row['generation']!=generation:raise RuntimeError('RGBD_RESET_GENERATION_CHANGED')
            generation=row['generation']
            if last_unique is not None and stamp<=last_unique:
                if stamp<last_unique:raise RuntimeError('REGRESSING_SENSOR_GENERATION_INPUT')
                time.sleep(.01);continue
            rgb,depth,K,T=render(model,data,renderer,camera,row)
            result,mask=estimate(rgb,depth,K,T)
            record={**result,'pass':True,'source_host_ns':stamp,'created_host_ns':time.monotonic_ns(),'generation':generation,'estimated_center_world_m':result['center_xy_world_m']+[result['top_world_z_m']-half_height_m],'height_prior_m':half_height_m*2,'object_truth_used_by_estimator':False,'classification':'FRESH_SIMULATED_CALIBRATED_HEAD_RGBD','camera_intrinsics':K,'T_world_optical':T.tolist()}
            try:qualify_estimate(record,generation=generation)
            except RuntimeError as exc:
                if not str(exc).startswith('STALE_RGBD_SOURCE'):raise
                # Discard delayed images; never refresh the last good stamp.
                # An active consumer still stops at its unchanged 0.20s age.
                # Only pre-motion readiness may await a subsequent good frame.
                dropped.append({'source_host_ns':stamp,'completed_host_ns':time.monotonic_ns(),'reason':str(exc),'published':False})
                last_unique=stamp;continue
            if not trace or len(trace)%10==0:
                Image.fromarray(rgb).save(output/f'rgb_{len(trace):04}.png');Image.fromarray(np.uint8(mask)*255).save(output/f'mask_{len(trace):04}.png')
            publish(record);trace.append(record);last_unique=stamp
            time.sleep(max(0.,.08-(time.monotonic()-tick)))
    except Exception as exc:
        publish({'pass':False,'failure':str(exc),'created_host_ns':time.monotonic_ns()})
        if 'rgb' in locals():
            Image.fromarray(rgb).save(output/'failure_rgb.png');np.save(output/'failure_depth_m.npy',depth)
            pg.write_new_text(output/'failure_sensor_input.json',json.dumps({'classification':'SENSOR_GENERATION_INPUT_FOR_OFFLINE_FAILURE_ANALYSIS_ONLY','row':row,'K':K,'T':T.tolist(),'failure':str(exc)})+'\n')
        raise
    finally:
        renderer.close();pg.write_new_text(output/'trace.json',json.dumps(trace)+'\n');pg.write_new_text(output/'dropped_frames.json',json.dumps(dropped)+'\n')
        # This subdirectory belongs to the parent's reserved evidence root;
        # the parent writes the recursive artifact manifest after teardown.

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path)
    p.add_argument('--stream-scene',type=Path);p.add_argument('--native-latest',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--half-height-m',type=float,default=.15);p.add_argument('--host-model-cache',type=Path)
    a=p.parse_args()
    if a.stream_scene:
        if not a.native_latest or not a.output_dir:raise ValueError('native state and unique output required')
        stream(a.stream_scene,a.native_latest,a.output_dir,half_height_m=a.half_height_m,cache=a.host_model_cache);return
    if not a.capture:raise ValueError('capture or stream mode required')
    record=json.loads((a.capture/'result.json').read_text());plan=json.loads(Path(record['hal_arm_runtime_replan']['path']).read_text());state=json.loads(Path(plan['standing_state']).read_text())
    stamp=state['native_snapshot']['host_monotonic_ns']
    with (a.capture/'native_physics.jsonl').open() as f:row=min((json.loads(line) for line in f),key=lambda r:abs(r['host_monotonic_ns']-stamp))
    scene=Path(plan['scene']['path'])
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=plan['scene']['sha256']:raise ValueError('scene hash mismatch')
    out=pg.create_unique_output_directory(Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46'),prefix='native-rgbd-recorded-sensor-preflight').native_path
    try:
        result=capture_once(scene,row,out);result['capture_mode']='SAVED_NATIVE_POSE_REPLAY_NOT_FRESH_LIVE'
        # Evaluation is strictly after estimator completion; never selects it.
        truth=row['bodies']['test_cylinder'];result['evaluation_only']={'center_xy_error_m':float(np.linalg.norm(np.array(result['center_xy_world_m'])-truth['position_m'][:2])),'top_error_m':float(result['top_world_z_m']-(truth['position_m'][2]+.15)),'diameter_error_m':float(result['diameter_m']-.07)}
        pg.write_new_text(out/'result.json',json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(out),**result},indent=2))
    except Exception as e:
        pg.write_new_text(out/'failure.json',json.dumps({'type':type(e).__name__,'message':str(e)})+'\n');raise
    finally:pg.write_artifact_manifest(out)

if __name__=='__main__':main()
