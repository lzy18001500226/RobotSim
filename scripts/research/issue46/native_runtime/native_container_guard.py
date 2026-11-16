"""Read-only X2 exclusivity check; unrelated isolated MoSim jobs are preserved."""
import json
import subprocess


def require_exclusive_x2_runtime():
    ids=subprocess.check_output(["docker","ps","-q"],text=True,timeout=5).split()
    if not ids:return []
    records=json.loads(subprocess.check_output(["docker","inspect",*ids],text=True,timeout=5))
    preserved=[]
    for r in records:
        config=r.get("Config",{});host=r.get("HostConfig",{})
        labels=config.get("Labels") or {}
        # This specific independently owned ROS1/tools compose project uses a
        # separate bridge and no X2 assets. Unknown containers remain blocked.
        mounts=r.get("Mounts",[])
        isolated_mosim=(labels.get("com.docker.compose.project")=="mosim"
            and config.get("Image","").startswith(("mosim/","mosim:"))
            and "ROS_DISTRO=noetic" in config.get("Env",[])
            and host.get("NetworkMode") in ("mosim_default","none")
            and not host.get("Privileged",False)
            and host.get("PidMode","")!="host" and host.get("IpcMode","")!="host"
            and bool(mounts) and all(
                (m.get("Destination","").startswith("/workspace/MoSim")
                 and "robotsim" not in m.get("Source","").lower())
                or (m.get("Destination") in ("/tmp/.X11-unix","/usr/lib/wsl") and m.get("RW") is False)
                for m in mounts))
        if not isolated_mosim:
            raise RuntimeError("conflicting or unverified container:"+r.get("Name",r.get("Id","UNKNOWN")))
        preserved.append(dict(name=r["Name"],image=config["Image"],
                              network=host["NetworkMode"],action="UNTOUCHED"))
    return preserved
