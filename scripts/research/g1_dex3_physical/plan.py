from baseline import *
import cv2

def render(m,d,p,target=None,dist=.55,az=115,el=-25):
    cam=mj.MjvCamera();cam.lookat[:]=target if target is not None else d.xpos[bid(m,'right_wrist_yaw_link')]+[.1,0,0]
    cam.distance=dist;cam.azimuth=az;cam.elevation=el
    with mj.Renderer(m,height=720,width=960) as r:
        opt=mj.MjvOption();opt.geomgroup[0]=True;opt.geomgroup[1]=True
        r.update_scene(d,camera=cam,scene_option=opt);rgb=r.render()
        path_governance.validate_output_path(p,require_new=True)
        if not cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f'failed to write render: {p}')
        path_governance.validate_output_path(p,require_exists=True,expected_kind='file')

def geom_for(m,body):return next(g for g in range(m.ngeom) if m.geom_bodyid[g]==bid(m,body) and (m.geom_contype[g] or m.geom_conaffinity[g]))
def dist(m,d,a,b):
    seg=np.zeros(6);x=mj.mj_geomDistance(m,d,a,b,1.,seg);return x,seg

def plan_hand(m,d,output_dir):
    qa,va=indices(m,HAND);w=bid(m,'right_wrist_yaw_link');b=bid(m,'green_box');bj=jid(m,'bottle_free');bq=m.jnt_qposadr[bj]
    gl=[geom_for(m,'right_hand_'+s+'_link') for s in ['thumb_2','index_1','middle_1']]
    bodyg=gid(m,'bottle_body');wpos=d.xpos[w].copy();R=d.xmat[w].reshape(3,3).copy()
    allg=[g for g in range(m.ngeom) if 'right_hand' in name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    hand_limits=m.jnt_range[[jid(m,n) for n in HAND]]
    # Bottle local z aligns hand separation axis. Only initialization states here.
    def place(x):
        d.qpos[qa]=x[:7];d.qpos[bq:bq+3]=wpos+R@np.r_[x[7:9],.04]
        quat=Rotation.from_matrix(R).as_quat();d.qpos[bq+3:bq+7]=quat[[3,0,1,2]];mj.mj_forward(m,d)
    def fun(x):
        place(x);ds=[dist(m,d,g,bodyg)[0] for g in gl]
        cs=[dist(m,d,g,bodyg)[0] for g in allg if g not in gl]
        out=[v*1000 for v in ds]
        out+= [min(0,v-.001)*1000 for v in cs]
        out+= [0.1*(x[0]),0.1*(x[3]-x[5]),0.1*(x[4]-x[6])]
        # Maximize opposing contact normals, not just count pairs.
        pts=[dist(m,d,g,bodyg)[1][3:] for g in gl]
        center=wpos+R@np.r_[x[7:9],0.]
        norms=[(R.T@(p-center))[:2]/max(np.linalg.norm((R.T@(p-center))[:2]),1e-9) for p in pts]
        out+=list((norms[0]+(norms[1]+norms[2])/2)*3)
        return out
    x0=np.r_[[0.,-.3,-.8,.8,.8,.8,.8],.15,.06]
    res=least_squares(fun,x0,bounds=(np.r_[hand_limits[:,0]+.015,.11,.035],np.r_[hand_limits[:,1]-.015,.20,.115]),diff_step=1e-4,max_nfev=200,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    place(res.x)
    result={'q':dict(zip(HAND,res.x[:7])),'bottle_center_local_body':[*res.x[7:9],0.], 'bottle_origin_local':[*res.x[7:9],.04], 'cost':res.cost,'nfev':res.nfev,
       'contacts':[{'body':name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]),'distance':dist(m,d,g,bodyg)[0],'points':dist(m,d,g,bodyg)[1].tolist()} for g in allg]}
    result.update({'evidence_directory':str(output_dir),
       'evidence_directory_paths':path_governance.describe_output_path(output_dir),
       'artifact_manifest':path_governance.describe_output_path(output_dir/'artifact_paths.json')})
    path_governance.write_new_text(output_dir/'hand_geometry_plan.json',json.dumps(result,indent=2)+'\n')
    render(m,d,output_dir/'planned_hand.png',wpos+R@np.r_[res.x[7:9],0],.45)
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    output_dir=reserve_output_run('plan')
    m=build(output_dir/'fixed_base_model.xml');d=init(m);result=plan_hand(m,d,output_dir)
    manifest=path_governance.write_artifact_manifest(output_dir)
    print(json.dumps({'plan':result,'artifact_manifest':manifest['manifest']},indent=2))
