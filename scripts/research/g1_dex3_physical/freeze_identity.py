"""Freeze model identity and compare official control abstractions without more physics."""
from episode import *
import platform
m=mj.MjModel.from_xml_path(str(OUT/'dex3-contact-attempt02/model.xml'))
rows=[]
for j in range(m.njnt):
 n=name(m,mj.mjtObj.mjOBJ_JOINT,j)
 if n not in SOURCE_LIMITS:continue
 ai=int(np.where(m.actuator_trnid[:,0]==j)[0][0])
 rows.append({'name':n,'body':name(m,mj.mjtObj.mjOBJ_BODY,m.jnt_bodyid[j]),'axis_local':m.jnt_axis[j].tolist(),'MJCF_range_rad':m.jnt_range[j].tolist(),'URDF_limits':SOURCE_LIMITS[n],
    'actuator_id':ai,'actuator_name':name(m,mj.mjtObj.mjOBJ_ACTUATOR,ai),'gainprm':m.actuator_gainprm[ai].tolist(),'biasprm':m.actuator_biasprm[ai].tolist(),'force_range_Nm':m.actuator_forcerange[ai].tolist()})
meshes=[]
for mesh in ET.parse(SOURCE).getroot().findall('./asset/mesh'):
 p=SOURCE.parent/'meshes'/mesh.get('file');meshes.append({'name':mesh.get('name'),'path':str(p),'sha256':sha(p),'source_scale':mesh.get('scale','1 1 1')})
manifest={'runtime':{'Python':platform.python_version(),'MuJoCo':mj.__version__,'numpy':np.__version__},'official_model':{'repository':'unitreerobotics/unitree_ros','revision':SOURCE_REV,'XML':str(SOURCE),'XML_sha256':sha(SOURCE),'URDF':str(URDF),'URDF_sha256':sha(URDF),'license':'BSD-3-Clause'},
 'mounts':{'right':{'parent':'right_wrist_yaw_link','palm_mesh_xyz':[.0415,-.003,0],'palm_mesh_quat':[1,0,0,0]},'left':{'parent':'left_wrist_yaw_link','palm_mesh_xyz':[.0415,.003,0],'palm_mesh_quat':[1,0,0,0]}},
 'joints':rows,'meshes':meshes,'equality_count':m.neq,'mocap_count':m.nmocap,
 'simulation_derived':['fixed pelvis, no object constraint','0.5ms implicitfast','SDK-form PD implemented as implicit affine general actuators','body kp120 kd5 with joint gravity compensation; hand kp1.5 kd.1','table translated to [0.55,-0.30,0]; bottle initial [0.38,-0.22,0.9175]','bounded source-valid IK/contact planning and smooth trajectory','bilateral hand geometry retained; left hand held clear'],
 'source_not_changed':['all mesh coordinates/scales','all link masses/inertials','all wrist mounts','all joint axes and MJCF hard ranges','all collision shapes and contact filters','bottle 0.570kg and friction 1.4 / .02 / .001','table shape/friction'],
 'source_format_discrepancies':json.loads((OUT/'source_range_discrepancies.json').read_text()),
 'hardware_order_per_hand':['thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1'],
 'mapping_note':'Motor IDs are resolved by exact joint names. MJCF declaration/actuator order is not assumed to be hardware ordering.',
 'official_reuse':{'SDK':'7 independent torque/position/velocity/kp/kd channels; grip demo kp1.5 kd0.1. Demo uses midpoints, no bottle pickup controller.',
 'IsaacLab':'fixed-base cylinder workflow and named Dex3 channels reused conceptually; task is joint-position action + DDS/teleop. Its preset hands kp100 kd10 armature0.1, effort300Nm velocity100rad/s and disabled self-collision are NOT adopted.',
 'MuJoCo':'official torque-PD bridge/model source; not proof of autonomous Dex3 bottle pickup. No Isaac runtime/ROS installed.'}}
(OUT/'model_control_provenance.json').write_text(json.dumps(manifest,indent=2))
for rid in ['dex3-contact-attempt01c','dex3-contact-attempt02']:
 r=json.loads((OUT/rid/'result.json').read_text());summary=json.loads((OUT/rid/'measured_summary.json').read_text());audit=json.loads((OUT/rid/'offline_audit.json').read_text());metrics=summary['joint_metrics']
 handv=max((v['peak_qvel'],n) for n,v in metrics.items())
 print(rid,'handv',handv,'effort',max((v['peak_effort'],n) for n,v in metrics.items()),'pos_violation',max(audit['sampled_URDF_range_violation_rad'].values()))
print('source meshes',len(meshes))
