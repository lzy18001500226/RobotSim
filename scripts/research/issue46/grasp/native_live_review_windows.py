"""Read-only live MuJoCo display for the native WSL physics process."""
import sys,os,json,time,math
from pathlib import Path
PACKAGES=Path(r"C:\Users\HP\.codex\visualizations\2026\10\03\01a1014f-2b35-7fe2-b6f6-a934b993405b\mujoco336-review-runtime\packages")
sys.path.insert(0,str(PACKAGES))
import numpy as np,mujoco,glfw
ROOT=Path(sys.argv[1]);CACHE=Path(sys.argv[2]);REPO=Path(sys.argv[3]);sys.path.insert(1,str(REPO))
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame,digest
def host(value):
    value=str(value)
    return "\\\\wsl.localhost\\Ubuntu-22.04"+value.replace("/",chr(92)) if value.startswith("/home/") else value
manifest=json.loads((ROOT/"live_view_scene.json").read_text());binary=json.loads(CACHE.read_text())
if manifest["scene"]["sha256"]!=binary["source_scene"]["sha256"]:raise RuntimeError("LIVE_VIEW_SCENE_IDENTITY_MISMATCH")
binary_path=Path(host(binary["path"]))
if digest(binary_path)!=binary["sha256"]:raise RuntimeError("LIVE_VIEW_BINARY_HASH_MISMATCH")
model=mujoco.MjModel.from_binary_path(str(binary_path));data=mujoco.MjData(model)
if not glfw.init():raise RuntimeError("live GLFW init failed")
window=glfw.create_window(1280,850,"MuJoCo LIVE - WSL physics",None,None)
if not window:raise RuntimeError("live GLFW window failed")
glfw.set_window_pos(window,60,80);glfw.make_context_current(window);glfw.swap_interval(1)
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[.95,-1.5,.80];cam.distance=1.9;cam.azimuth=135.;cam.elevation=-20.
opt=mujoco.MjvOption();opt.geomgroup[3:]=0
scene=mujoco.MjvScene(model,maxgeom=20000);context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_150)
state={"cursor":None,"follow":True,"phase":"","phase_read_at":0.};stamp=None;receipt=None;row=None;ready_written=False;ended=False;viewer_error=None
def cursor(w,x,y):
    before=state["cursor"];state["cursor"]=(x,y)
    if before is None:return
    height=max(1,glfw.get_window_size(w)[1])
    if glfw.get_mouse_button(w,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif glfw.get_mouse_button(w,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_MOVE_V;state["follow"]=False
    else:return
    mujoco.mjv_moveCamera(model,action,(x-before[0])/height,(y-before[1])/height,scene,cam)
def scroll(w,x,y):mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,scene,cam)
def key(w,k,s,a,m):
    if a!=glfw.PRESS:return
    if k==glfw.KEY_ESCAPE:glfw.set_window_should_close(w,True)
    elif k==glfw.KEY_5 and row is not None:
        state["follow"]=True;cam.lookat[:]=np.asarray(row["bodies"]["pelvis"]["position_m"])+[0,0,.2];cam.distance=1.9
    elif k==glfw.KEY_6 and row is not None:
        state["follow"]=False;cam.lookat[:]=row["bodies"]["R_omnipicker_base_link"]["position_m"];cam.distance=.65
glfw.set_cursor_pos_callback(window,cursor);glfw.set_scroll_callback(window,scroll);glfw.set_key_callback(window,key)
def atomic(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value));os.replace(tmp,path)
try:
    while not glfw.window_should_close(window):
        began=time.monotonic()
        if ended:glfw.wait_events()
        else:glfw.poll_events()
        before_stamp=stamp
        try:
            fresh=row if ended else json.loads((ROOT/"live_view_latest.json").read_text())
            if fresh is not None and fresh["generation"]>=1 and (stamp is None or fresh["host_monotonic_ns"]>stamp):
                row=fresh;stamp=fresh["host_monotonic_ns"];receipt=time.monotonic()
        except (OSError,ValueError,KeyError):pass
        ended=(ROOT/"live_review_finished.json").exists()
        if row is not None and stamp!=before_stamp:apply_recorded_native_frame(model,data,row,visual_kinematics_only=True)
        if row is not None and state["follow"]:cam.lookat[:]=np.asarray(row["bodies"]["pelvis"]["position_m"])+[0,0,.2]
        mujoco.mjv_updateScene(model,data,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene)
        if row is None:scene.ngeom=0
        width,height=glfw.get_framebuffer_size(window);viewport=mujoco.MjrRect(0,0,width,height)
        mujoco.mjr_render(viewport,scene,context)
        mode="RUN ENDED - LAST ACTUAL FRAME" if ended else "LIVE WSL NATIVE PHYSICS" if row is not None and time.monotonic()-receipt<=.2 else "WAITING FOR FRESH NATIVE FRAME"
        detail="5 Body | 6 Gripper | mouse rotate / right-pan / wheel-zoom | ESC closes review and requests stop"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,mode,detail,context)
        if row:
            p=row["bodies"]["pelvis"]["position_m"]
            if not ended and began-state["phase_read_at"]>.5:
                try:state["phase"]=json.loads((ROOT/"native_tracking_phase.json").read_text()).get("phase","")
                except (OSError,ValueError):pass
                state["phase_read_at"]=began
            phase=state["phase"]
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,f"sim={row['sim_time_s']:.3f}s | pelvis=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) | {phase}","Actual WSL MuJoCo+MC states. This window has no physics or command publisher.",context)
        glfw.swap_buffers(window)
        if row and not ready_written:
            atomic(ROOT/"live_view_ready.json",dict(viewer_pid=os.getpid(),source_pid=row["process_pid"],generation=row["generation"],source_host_ns=stamp,scene_sha256=manifest["scene"]["sha256"],rendered=True))
            ready_written=True
        time.sleep(max(0,.02-(time.monotonic()-began)))
except BaseException as exc:
    viewer_error=repr(exc)
    raise
finally:
    atomic(ROOT/"live_view_closed.json",dict(viewer_pid=os.getpid(),normal_close=viewer_error is None,error=viewer_error,physics_steps=0))
    context.free();glfw.destroy_window(window);glfw.terminate()
