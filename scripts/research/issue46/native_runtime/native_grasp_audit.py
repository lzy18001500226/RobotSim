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
