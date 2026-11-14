"""Read-only audit of saved X2 command/pose records; never imports ROS or MuJoCo.

Disposition: RETAIN_RESEARCH_DIAGNOSTIC. Commands are not motion predictions.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg

def project_displacement(start,end):
    values=[start[k] for k in ("x","y","yaw")]+[end[k] for k in ("x","y","yaw")]
    if not all(math.isfinite(v) for v in values):raise ValueError("nonfinite measured pose")
    dx,dy=end["x"]-start["x"],end["y"]-start["y"];c,s=math.cos(start["yaw"]),math.sin(start["yaw"])
    return {"world_dx_m":dx,"world_dy_m":dy,"initial_body_forward_m":c*dx+s*dy,
            "initial_body_left_m":-s*dx+c*dy,
            "yaw_change_rad":(end["yaw"]-start["yaw"]+math.pi)%(2*math.pi)-math.pi}

def message_summary(messages,*,dds=False):
    keys=("forward_velocity","lateral_velocity","angular_velocity") if dds else ("forward","lateral","angular")
    records=[(row.get("t"),(row.get("raw",{}) if dds else row)) for row in messages]
    nonzero=[(t,m) for t,m in records if any(abs(float(m.get(k,0.)))>1e-7 for k in keys)]
    result={"packets":len(records),"nonzero_packets":len(nonzero),"is_mc_receipt_ack":False}
    if not nonzero:return result
    result.update(first_nonzero=dict(time=nonzero[0][0],values={k:nonzero[0][1].get(k,0.) for k in keys}),
        nonzero_span_s=nonzero[-1][0]-nonzero[0][0],
        ranges={k:[min(float(m.get(k,0.)) for _,m in nonzero),max(float(m.get(k,0.)) for _,m in nonzero)] for k in keys},
        combined_translation_packets=sum(abs(m.get(keys[0],0.))>.005 and abs(m.get(keys[1],0.))>.005 for _,m in nonzero),
        combined_forward_yaw_packets=sum(abs(m.get(keys[0],0.))>.005 and abs(m.get(keys[2],0.))>.005 for _,m in nonzero))
    if dds:
        active=[m for t,m in records if nonzero[0][0]<=t<=nonzero[-1][0]]
        result["zero_packets_inside_nonzero_span"]=sum(all(abs(float(m.get(k,0.)))<=1e-7 for k in keys) for m in active)
    return result

def ideal_forward_s_curve(speed,yaw_rate,angle):
    if not all(math.isfinite(v) for v in (speed,yaw_rate,angle)) or speed<=0 or yaw_rate<=0 or not 0<angle<math.pi/2:raise ValueError("invalid ideal arc")
    radius=speed/yaw_rate
    return {"classification":"IDEAL_UNICYCLE_GEOMETRY_NOT_X2_PHYSICS","forward_m":2*radius*math.sin(angle),
            "left_m":2*radius*(1-math.cos(angle)),"final_yaw_rad":0.,"duration_s":2*angle/yaw_rate}

def audit(path):
    raw=path.read_bytes();data=json.loads(raw);trace=[row for row in data.get("trace",[]) if row.get("pose")]
    end=data.get("end_pose")
    if end is None and trace:end=trace[-1]["pose"]
    result={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),
            "safe_stop_flag":data.get("pass_result"),"failure":data.get("failure"),
            "pause":data.get("task_replan_pause"),"arrival_is_not_implied_by_safe_stop":True,
            "published":message_summary(data.get("commands",[])),
            "dds_observed":message_summary(data.get("bridge_commands",[]),dds=True)}
    if trace and end:
        result.update(start=trace[0]["pose"],end=end,displacement=project_displacement(trace[0]["pose"],end),
            heading_range_deg=[math.degrees(min(x["pose"]["yaw"] for x in trace)),math.degrees(max(x["pose"]["yaw"] for x in trace))],
            observed_modes=dict(Counter(str((x.get("common_mode_desc"),x.get("get_mode_desc"))) for x in trace)),
            observed_sources=dict(Counter(str(x.get("common_source",{}).get("name")) for x in trace)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--records",type=Path,nargs="+",required=True);a=p.parse_args();out=a.output_dir.resolve(strict=True)
    if not (out/"task.json").exists():raise ValueError("existing owned review reservation required")
    rows=[audit(x.resolve(strict=True)) for x in a.records]
    result={"scope":"RECORDED_INPUT_OUTPUT_AUDIT_NOT_NEW_MOTION_OR_CAUSAL_MODE_AB_TEST","records":rows,
            "ideal_forward_s_12cm":ideal_forward_s_curve(.12,.40,math.acos(.8)),"new_mj_step_calls":0,"ros_nodes_started":0}
    pg.write_new_text(out/"recorded_motion_audit.json",json.dumps(result,indent=2)+"\n")
    for row in rows:print(json.dumps({"record":row["path"],"displacement":row.get("displacement"),"dds":row["dds_observed"],"failure":row["failure"]}))
if __name__=="__main__":main()
