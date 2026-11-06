"""Build a read-only native step observer using exact-version MuJoCo headers."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, runtime_library):
    from scripts.agent import path_governance as pg
    output=Path(output);library=Path(runtime_library)
    api=ctypes.CDLL(str(library));api.mj_version.restype=ctypes.c_int
    version=api.mj_version()
    if version!=337:raise RuntimeError('native observer currently qualified only for engine ABI 337')
    # Header discovery is not a dependency install and never falls back to a
    # different ABI. Source package versions/hashes are persisted below.
    matches=[]
    for header in (Path.home()/'.cache/uv/archive-v0').glob('**/mujoco/include/mujoco/mujoco.h'):
        if '#define mjVERSION_HEADER 337' in header.read_text():matches.append(header)
    if not matches:raise RuntimeError('exact MuJoCo 3.3.7 headers absent; no native model will be launched')
    header=sorted(matches)[0];include=header.parents[1]
    source=Path(__file__).with_suffix('.c');target=output/'native_grasp_audit.so'
    if target.exists():raise FileExistsError(target)
    command=['gcc','-O2','-fPIC','-shared','-Wall','-Wextra','-Werror','-I',str(include),str(source),'-o',str(target),'-ldl','-pthread']
    r=subprocess.run(command,text=True,capture_output=True,timeout=20)
    pg.write_new_text(output/'native_audit_build.stdout',r.stdout)
    pg.write_new_text(output/'native_audit_build.stderr',r.stderr)
    if r.returncode:raise RuntimeError('native observer build failed: '+r.stderr)
    pg.write_new_bytes(output/source.name,source.read_bytes())
    headers={p.name:digest(p) for p in header.parent.glob('*.h')}
    record={'classification':'READ_ONLY_NATIVE_MJ_STEP_OBSERVER','engine_api_version':version,'runtime_library':str(library),'runtime_library_sha256':digest(library),'header_path':str(header),'header_sha256':headers,'source_sha256':digest(source),'binary_sha256':digest(target),'command':command,'writes_model_or_state':False,'scope':'post-mj_step snapshot; no extra mj_forward; timestamps and performance recorded; not grasp acceptance'}
    pg.write_new_text(output/'native_audit_provenance.json',json.dumps(record,indent=2)+'\n')
    return target


def launch_prefix():
    """Only the simulator process receives the read-only interposer."""
    return "env LD_PRELOAD=/workspace/evidence/native_grasp_audit.so ROBOTSIM_NATIVE_AUDIT_OUT=/workspace/evidence/native_physics.jsonl "


def require_fresh_sample(path, *, now_ns=None):
    import time
    import math
    raw = Path(path).read_bytes()[-1048576:]
    lines = raw.splitlines()
    if not raw.endswith(b"\n"):
        lines = lines[:-1]
    for line in reversed(lines):
        try:
            row = json.loads(line)
            break
        except (ValueError, UnicodeDecodeError):
            continue
    else:
        raise RuntimeError("NATIVE_OBSERVER_NO_COMPLETE_SAMPLE")
    age = ((time.monotonic_ns() if now_ns is None else now_ns) - row["host_monotonic_ns"]) * 1e-9
    if not math.isfinite(age) or not 0.0 <= age <= 0.20:
        raise RuntimeError("NATIVE_OBSERVER_STALE_SAMPLE")
    if row.get("engine_version") != 337 or any(row.get("bodies", {}).get(n) is None for n in ("pelvis", "R_omnipicker_base_link", "test_cylinder")):
        raise RuntimeError("NATIVE_OBSERVER_MISSING_REQUIRED_BODIES")
    return {"pass": True, "age_s": age, "engine_version": row["engine_version"], "sim_time_s": row["sim_time_s"], "generation": row["generation"]}


def planning_state_from_native(row, previous):
    """One coherent actual-engine sample for offline planning; never writes runtime."""
    import copy
    import math
    from scipy.spatial.transform import Rotation
    state = copy.deepcopy(previous)
    body = row["bodies"]["pelvis"]
    q = Rotation.from_matrix([body["rotation"][i:i+3] for i in (0,3,6)]).as_quat()
    state["base_pose_world"] = {"xyz_m": list(body["position_m"]), "quaternion_wxyz": [float(q[3]), *map(float,q[:3])]}
    for group in ("arm_positions_rad", "leg_positions_rad", "waist_positions_rad"):
        for name in state[group]:
            value = float(row["joints"][name]["q"])
            if not math.isfinite(value):
                raise RuntimeError("NONFINITE_NATIVE_PLANNING_STATE:" + name)
            state[group][name] = value
    state["native_snapshot"] = {k:row[k] for k in ("host_monotonic_ns", "sim_time_s", "generation")}
    state["state_source"] = "SYNCHRONIZED_NATIVE_MJDATA_NOT_ROS_CACHE"
    return state


def coupled_opening_budget(model, root_joint, actuator, q):
    """Quasi-static estimate, not proof of solver or hardware feasibility."""
    import mujoco
    import math
    if not math.isfinite(q):
        raise ValueError("nonfinite root position")
    terms=[{"joint": mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,root_joint), "derivative": 1., "frictionloss_nm": float(model.dof_frictionloss[model.jnt_dofadr[root_joint]])}]
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[i] != root_joint:
            continue
        j=int(model.eq_obj1id[i])
        derivative=sum(k*float(model.eq_data[i,k])*q**(k-1) for k in range(1,5))
        terms.append({"joint":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),"derivative":derivative,"frictionloss_nm":float(model.dof_frictionloss[model.jnt_dofadr[j]])})
    resistance=sum(abs(t["derivative"])*t["frictionloss_nm"] for t in terms)
    kp=float(model.actuator_gainprm[actuator,0]);gear=float(model.actuator_gear[actuator,0])
    if kp<=0 or gear!=-1 or abs(float(model.actuator_biasprm[actuator,1])+kp)>1e-9:
        raise ValueError("budget requires the recorded gear=-1 position servo")
    command=-q+resistance/kp
    return {"q_rad":q,"estimated_coupled_coulomb_nm":resistance,"terms":terms,"kp":kp,"opening_command_to_balance_coulomb_estimate":command,"command_upper_bound":float(model.actuator_ctrlrange[actuator,1]),"estimate_within_command_range":command<=float(model.actuator_ctrlrange[actuator,1]),"classification":"QUASISTATIC_ESTIMATE_NOT_PHYSICS_ACCEPTANCE"}
