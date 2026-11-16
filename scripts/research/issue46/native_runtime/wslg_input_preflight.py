"""Read-only WSLg pointer preflight for the live human-review entry point.

No simulated key/button events, compositor restart, physics or MC requests.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime.native_mujoco_reset import X11


def classify(windows, x11_mask):
    required=("LeftDown","MiddleDown","RightDown")
    if any(type(windows.get(k)) is not bool for k in required):
        return "WINDOWS_BUTTON_STATE_UNVERIFIED"
    if any(windows[k] for k in required):
        return "RELEASE_PHYSICAL_MOUSE_BUTTONS_BEFORE_AUTOMATION"
    if x11_mask&(256|512|1024):
        return "STALE_X11_POINTER_BUTTON_STATE"
    return "PASS"


def main():
    # Do not interrupt a currently open experiment or interactive review.
    running=subprocess.run(["docker","ps","--format","{{.Names}}"],text=True,capture_output=True,timeout=5)
    if running.returncode:raise RuntimeError("Docker availability not verified")
    if running.stdout.strip():raise RuntimeError("已有实验或审核窗口，请先正常结束：\n"+running.stdout)
    powershell=Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    helper=Path(__file__).with_name("wslg_mujoco_foreground.ps1")
    path=subprocess.check_output(["wslpath","-w",str(helper)],text=True,timeout=3).strip()
    result=subprocess.run([str(powershell),"-NoProfile","-ExecutionPolicy","Bypass","-WindowStyle","Hidden","-File",path,"-ProbeOnly"],text=True,capture_output=True,timeout=6)
    if result.returncode:raise RuntimeError("Windows pointer probe failed:"+result.stdout+result.stderr)
    windows=json.loads(result.stdout.strip().splitlines()[-1])
    if windows.get("status")!="READ_ONLY_POINTER_PROBE" or windows.get("mouseEvents")!=0 or windows.get("keyboardEvents")!=0:
        raise RuntimeError("Windows pointer evidence invalid")
    x=X11()
    root,child=ctypes.c_ulong(),ctypes.c_ulong()
    rx,ry,wx,wy=ctypes.c_int(),ctypes.c_int(),ctypes.c_int(),ctypes.c_int()
    mask=ctypes.c_uint()
    if not x.lib.XQueryPointer(x.display,x.root,ctypes.byref(root),ctypes.byref(child),ctypes.byref(rx),ctypes.byref(ry),ctypes.byref(wx),ctypes.byref(wy),ctypes.byref(mask)):
        raise RuntimeError("X11 pointer query unavailable")
    verdict=classify(windows,mask.value)
    out=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="live-review-pointer-preflight").native_path
    record=dict(classification=verdict,windows=windows,x11_mask=mask.value,
                source_monotonic_ns=time.monotonic_ns(),input_events_sent=0,
                physics_started=False,display=os.environ.get("DISPLAY"))
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n")
    pg.write_artifact_manifest(out)
    print(json.dumps(dict(pointer_preflight=verdict,evidence=str(out)),ensure_ascii=False),flush=True)
    if verdict!="PASS":
        raise RuntimeError("窗口输入未就绪，尚未启动仿真："+verdict+
            "。请先松开鼠标；若 Windows 已松开但 Linux 仍按下，需恢复 WSLg 输入。不要反复启动抓取或关闭整个 WSL。")


if __name__=="__main__":main()
