from baseline import *
from plan import render
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for rid in ['dex3-contact-attempt01c','dex3-contact-attempt02']:
 out=OUT/rid;rr=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()];m=mj.MjModel.from_xml_path(str(out/'model.xml'));d=mj.MjData(m)
 qa,va=indices(m,rr[0]['joint_names']);bqa=m.jnt_qposadr[jid(m,'bottle_free')]
 last=rr[-1];d.qpos[qa]=last['q'];d.qpos[bqa:bqa+3]=last['bottle_pos'];d.qpos[bqa+3:bqa+7]=last['bottle_quat'];mj.mj_forward(m,d)
 render(m,d,out/'overview_final.png',[.2,-.15,.85],2.3,135,-18)
 render(m,d,out/'contact_front.png',[.38,-.22,.91],.55,90,-10)
 render(m,d,out/'contact_side.png',[.38,-.22,.91],.55,0,-10)
 t=np.array([r['time'] for r in rr]);fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
 axes[0].plot(t,[r['hand_z'] for r in rr],label='measured right-hand upward force');axes[0].plot(t,[r['table_z'] for r in rr],label='table reaction');axes[0].axhline(.57*9.81,color='k',ls='--',label='bottle weight 5.5917N');axes[0].set_ylabel('Force [N]');axes[0].legend()
 b=np.array([r['bottle_pos'] for r in rr]);w=np.array([r['wrist_pos'] for r in rr]);start=next(i for i,r in enumerate(rr) if r['stage']=='LOAD_TRANSFER')
 axes[1].plot(t,(w[:,2]-w[start,2])*1000,label='wrist z relative to LOAD start');axes[1].plot(t,(b[:,2]-b[start,2])*1000,label='bottle z relative to LOAD start');axes[1].set_ylim(-2,8);axes[1].set_ylabel('Displacement [mm]');axes[1].legend()
 for link in ['thumb_2','index_1','middle_1']:
  ns=[sum(c['contact_force'][0] for c in r['contacts'] if c['other']=='right_hand_'+link+'_link') for r in rr]
  axes[2].plot(t,ns,label=link)
 axes[2].set_ylabel('Normal force [N]');axes[2].set_xlabel('simulation time [s]');axes[2].legend();axes[0].set_title(rid+' — LOAD_READY FAIL, no airborne lift')
 for ax in axes:ax.grid();ax.axvline(10,color='gray',ls=':')
 fig.tight_layout();fig.savefig(out/'forces_and_displacement.png',dpi=150);plt.close(fig)
print('postprocess complete; physics steps = 0')
