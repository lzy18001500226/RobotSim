#!/usr/bin/env python3
"""Bounded 24-pose full-X2 PR90 static grasp screen; physics stepping is prohibited."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,struct,subprocess,sys,uuid,xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"/"research"))
import issue46_manipulation_integration.audit_pr90_full_x2_static as base
import issue46_manipulation_integration.audit_pr90_bounded_pose_variants as bounded
pg=base.pg
REV=base.REV
PR90_HEAD="facdaa0d6e5f8152168d2e812d864b660eac9f96"
PR90_MODEL_SHA="32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45"
OLD_MODEL_SHA="0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01"
CLEARANCE_MM=2.0
CONTACT_TOL_MM=0.1
CONTACT_BAND_HALF_MM=5.0
MAX_SWEEP_DEPTH=7
MAX_REFINED=12
ARM=tuple(base.ARM);HAND=tuple(base.HAND)
SOURCE_BODIES={"R_omnipicker_base_link":"base_link.stl","R_hand_narrow1_Link":"narrow1_Link.stl",
 "R_hand_narrow2_Link":"narrow2_Link.stl","R_hand_narrow3_Link":"narrow3_Link.stl",
 "R_hand_narrow_loop_Link":"narrow_loop_Link.stl","R_hand_wide1_Link":"wide1_Link.stl",
 "R_hand_wide2_Link":"wide2_Link.stl","R_hand_wide3_Link":"wide3_Link.stl",
 "R_hand_wide_loop_Link":"wide_loop_Link.stl"}
PAD_BODIES={"R_hand_narrow3_Link","R_hand_wide3_Link"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gn(m,g):return base.nm(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))
def bn(m,b):return base.body(m,int(b))
def jid(m,n):return base.jid(m,n)
def gid(m,n):return base.gid(m,n)
def ckey(a,b):return tuple(sorted((int(a),int(b))))

@dataclass
class Shape:
 points:np.ndarray
 normals:np.ndarray
 edges:np.ndarray

def hull_shape(points):
 p=np.unique(np.asarray(points,dtype=np.float64),axis=0)
 if len(p)<4:raise ValueError("active collision hull has <4 points")
 h=ConvexHull(p);es=set()
 for f in h.simplices:
  for i,j in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):es.add(tuple(sorted((int(i),int(j)))))
 es=sorted(es);e=p[[x[0] for x in es]]-p[[x[1] for x in es]]
 n=np.asarray(h.equations[:,:3],dtype=float);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-15)
 return Shape(p,n,e)

def ico():
 ph=(1+math.sqrt(5))/2;v=[]
 for a in (-1.,1.):
  for b in (-ph,ph):v.extend(((0.,a,b),(a,b,0.),(b,0.,a)))
 v=np.unique(np.asarray(v,float),axis=0);return v/np.linalg.norm(v[0])

def primitive_points(m,g,ncyl=96):
 t=int(m.geom_type[g]);s=np.asarray(m.geom_size[g],float)
 if t==int(mujoco.mjtGeom.mjGEOM_BOX):
  x,y,z=s[:3];return np.asarray([(a*x,b*y,c*z) for a in (-1,1) for b in (-1,1) for c in (-1,1)])
 if t==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
  r,h=float(s[0]),float(s[1]);rr=r/math.cos(math.pi/ncyl);a=np.arange(ncyl)*2*math.pi/ncyl
  ring=np.column_stack((rr*np.cos(a),rr*np.sin(a)))
  return np.vstack((np.column_stack((ring,np.full(ncyl,-h))),np.column_stack((ring,np.full(ncyl,h)))))
 ratio=math.sqrt(3)*(3+math.sqrt(5))/(3*math.sqrt(10+2*math.sqrt(5)))
 if t==int(mujoco.mjtGeom.mjGEOM_SPHERE):return ico()*(float(s[0])/ratio)
 if t==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):return ico()*(s[:3]/ratio)
 if t==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
  v=ico()*(float(s[0])/ratio);h=float(s[1])/2;return np.vstack((v+[0,0,-h],v+[0,0,h]))
 raise ValueError(f"unsupported active collision type {t}: {gn(m,g)}")

def sat_gap(a,pa,Ra,b,pb,Rb):
 # Project local vertex clouds in small batches; positive gaps are safe distance lower bounds.
 center=(np.mean(b.points,axis=0)@Rb.T+pb)-(np.mean(a.points,axis=0)@Ra.T+pa)
 groups=[a.normals@Ra.T,b.normals@Rb.T,np.eye(3)]
 if np.linalg.norm(center)>1e-12:groups.append((center/np.linalg.norm(center))[None,:])
 best=-1e30;best_axis=None
 for group in groups:
  for start in range(0,len(group),8):
   ax=np.asarray(group[start:start+8],dtype=float);nn=np.linalg.norm(ax,axis=1)
   ax=ax[nn>1e-10]/nn[nn>1e-10,None]
   if not len(ax):continue
   al=ax@Ra;bl=ax@Rb
   amin=np.min(a.points@al.T+pa@ax.T,axis=0);amax=np.max(a.points@al.T+pa@ax.T,axis=0)
   bmin=np.min(b.points@bl.T+pb@ax.T,axis=0);bmax=np.max(b.points@bl.T+pb@ax.T,axis=0)
   gaps=np.maximum(bmin-amax,amin-bmax);k=int(np.argmax(gaps))
   if float(gaps[k])>best:best=float(gaps[k]);best_axis=ax[k].tolist()
 return best,best_axis

def mesh_shape(m,mid):
 va,nv=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid]);fa,nf=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
 points=np.asarray(m.mesh_vert[va:va+nv],dtype=np.float64)
 faces=np.asarray(m.mesh_face[fa:fa+nf],dtype=np.int64)
 take=np.linspace(0,nf-1,min(256,nf),dtype=np.int64);tri=points[faces[take]]
 normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 lengths=np.linalg.norm(normals,axis=1);normals=normals[lengths>1e-12]/lengths[lengths>1e-12,None]
 canon=normals.copy();flip=canon[:,0]<-1e-10;canon[flip]*=-1
 flip=(np.abs(canon[:,0])<=1e-10)&(canon[:,1]<-1e-10);canon[flip]*=-1
 flip=(np.abs(canon[:,0])<=1e-10)&(np.abs(canon[:,1])<=1e-10)&(canon[:,2]<0);canon[flip]*=-1
 _,keep=np.unique(np.round(canon,6),axis=0,return_index=True)
 return Shape(points,normals[np.sort(keep)],np.empty((0,3),dtype=np.float64))

def aabb_gap(alo,ahi,blo,bhi):
 v=np.maximum(0.,np.maximum(alo-bhi,blo-ahi));return float(np.linalg.norm(v))

def self_test():
 a=hull_shape([(x*.05,y*.05,z*.05) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
 g,_=sat_gap(a,np.zeros(3),np.eye(3),a,np.array([.103,0,0]),np.eye(3))
 if not .0029<g<.0031:raise AssertionError("SAT separation regression failed")
 g,_=sat_gap(a,np.zeros(3),np.eye(3),a,np.array([.099,0,0]),np.eye(3))
 if g>0:raise AssertionError("SAT overlap regression failed")

class Evaluator:
 def __init__(self,m,d,root,obj,ob,oq,aq,ad,limits,friction):
  self.m,self.d,self.root=m,d,root;self.obj,self.ob,self.oq=int(obj),int(ob),int(oq)
  self.aq,self.ad,self.limits,self.friction=aq,ad,limits,friction
  self.cache={};self.shape={};self.geoms=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  for g in self.geoms:self.shape[g]=self._shape(g)
  self.local_bounds={g:(self.shape[g].points.min(0),self.shape[g].points.max(0)) for g in self.geoms if self.shape[g] is not None}
  self.arm_ids={n:jid(m,n) for n in ARM};self.hand_ids={n:jid(m,n) for n in HAND};self.excl=base.excl(root)
  self.reach={};self._derive_reach()
  self.robot=[g for g in self.geoms if int(m.geom_bodyid[g]) not in (0,self.ob) and not bn(m,int(m.geom_bodyid[g])).startswith("m0_table")]
  self.moving=[g for g in self.robot if bn(m,int(m.geom_bodyid[g])).startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker"))]
  self.scene=[g for g in self.geoms if int(m.geom_bodyid[g])==0 or bn(m,int(m.geom_bodyid[g])).startswith("m0_table")]
  ps=set()
  for g in self.robot:
   if base.coll(m,g,self.obj,self.excl):ps.add(tuple(sorted((g,self.obj))))
  for g in self.moving:
   for h in self.scene+self.robot:
    if base.coll(m,g,h,self.excl):ps.add(tuple(sorted((g,h))))
  self.path_pairs=sorted(p for p in ps if not self.support(*p))
  self.gripper=[g for g in self.moving if bn(m,int(m.geom_bodyid[g])).startswith(("R_hand_","R_omnipicker"))]
  self.gr_obj=sorted({tuple(sorted((g,self.obj))) for g in self.gripper if base.coll(m,g,self.obj,self.excl)})
  self.gr_self=sorted({tuple(sorted((g,h))) for i,g in enumerate(self.gripper) for h in self.gripper[i+1:] if base.coll(m,g,h,self.excl)})
  self.floor=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"floor"))
  if self.floor<0 or len(self.gripper)!=137 or len(self.gr_obj)!=len(self.gripper):raise ValueError("frozen scene/collision inventory mismatch")

 def _shape(self,g):
  if int(self.m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE):return None
  mid=int(self.m.geom_dataid[g])
  if mid>=0:
   if mid not in self.cache:self.cache[mid]=mesh_shape(self.m,mid)
   return self.cache[mid]
  return hull_shape(primitive_points(self.m,g))

 def _derive_reach(self):
  for g in self.geoms:
   sh=self.shape[g]
   if sh is None:continue
   gb=int(self.m.geom_bodyid[g]);chain=[];b=gb
   while b!=0:chain.append(b);b=int(self.m.body_parentid[b])
   chain=list(reversed(chain));self.reach[g]={};extent=float(np.max(np.linalg.norm(sh.points,axis=1)))
   for j in range(self.m.njnt):
    if self.m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:continue
    jb=int(self.m.jnt_bodyid[j])
    if jb not in chain:continue
    i=chain.index(jb);r=float(np.linalg.norm(self.d.xpos[jb]-self.d.xanchor[j]));parent=jb
    for child in chain[i+1:]:
     js=[k for k in range(int(self.m.body_jntadr[child]),int(self.m.body_jntadr[child]+self.m.body_jntnum[child])) if self.m.jnt_type[k]==mujoco.mjtJoint.mjJNT_HINGE]
     po=self.d.xpos[parent];co=self.d.xpos[child]
     if js:
      anchor=self.d.xanchor[js[0]];r+=float(np.linalg.norm(anchor-po)+np.linalg.norm(co-anchor))
      for other in js[1:]:r+=float(np.linalg.norm(self.d.xanchor[other]-anchor))
     else:r+=float(np.linalg.norm(co-po))
     parent=child
    r+=float(np.linalg.norm(self.d.geom_xpos[g]-self.d.xpos[gb]))+extent
    self.reach[g][j]=r
  self.max_reach=max((v for x in self.reach.values() for v in x.values()),default=0.)

 def apply(self,arm,hand):
  base.state(self.m,self.d,tuple(map(float,arm)),hand,self.oq,self.aq,self.obj,self.ob,self.friction)

 def world(self,g):
  sh=self.shape[g]
  if sh is None:return None
  R=self.d.geom_xmat[g].reshape(3,3).astype(float);p=self.d.geom_xpos[g].astype(float)
  lo,hi=self.local_bounds[g]
  corners=np.asarray([(x,y,z) for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
  wc=corners@R.T+p
  return (wc.min(0),wc.max(0),p,R)

 def support(self,a,b):
  names={gn(self.m,a),gn(self.m,b)};bodies={bn(self.m,int(self.m.geom_bodyid[a])),bn(self.m,int(self.m.geom_bodyid[b]))}
  return "floor" in names and bool(bodies&{"left_ankle_roll_link","right_ankle_roll_link"})

 def contacts(self):
  o={}
  for c in self.d.contact[:self.d.ncon]:o.setdefault(ckey(c.geom1,c.geom2),[]).append(c)
  return o

 def pair_gap(self,a,b,w,con):
  plane=int(mujoco.mjtGeom.mjGEOM_PLANE)
  if int(self.m.geom_type[a])==plane:
   n=self.d.geom_xmat[a].reshape(3,3)[:,2];p=self.d.geom_xpos[a]
   local=self.d.geom_xmat[b].reshape(3,3).T@n
   return float(np.min(self.shape[b].points@local)+self.d.geom_xpos[b]@n-p@n),None
  if int(self.m.geom_type[b])==plane:
   n=self.d.geom_xmat[b].reshape(3,3)[:,2];p=self.d.geom_xpos[b]
   local=self.d.geom_xmat[a].reshape(3,3).T@n
   return float(np.min(self.shape[a].points@local)+self.d.geom_xpos[a]@n-p@n),None
  wa,wb=w[a],w[b];bg=aabb_gap(wa[0],wa[1],wb[0],wb[1])
  if bg>=CLEARANCE_MM/1000:return bg,None
  g,axis=sat_gap(self.shape[a],wa[2],wa[3],self.shape[b],wb[2],wb[3])
  if g>1e-10:return g,axis
  cs=con.get(ckey(a,b),[])
  return (min(min(float(x.dist) for x in cs),0.) if cs else -1e-6),axis

 def bounds(self,a0,h0,a1,h1):
  ds={self.arm_ids[n]:abs(float(b)-float(a)) for n,a,b in zip(ARM,a0,a1)}
  ds.update({self.hand_ids[n]:abs(float(h1[n])-float(h0[n])) for n in HAND})
  return {g:sum(self.reach.get(g,{}).get(j,0.)*v/2 for j,v in ds.items()) for g in self.geoms}

 def eval_current(self,arm,hand,pairs,bounds=None,details=False):
  if pairs is None:pairs=self.path_pairs
  if not pairs:return {"pass":True,"minimum_clearance_mm":None,"minimum_swept_clearance_mm":None,
                      "worst_pair":None,"needs_subdivision":[],"near_pairs":[]}
  ids=set(x for p in pairs for x in p);w={g:self.world(g) for g in ids};con=self.contacts()
  worst=None;worstv=float("inf");needs=[];rows=[]
  plane=int(mujoco.mjtGeom.mjGEOM_PLANE);planes=[];regular=[]
  for a,b in pairs:
   if self.support(a,b):continue
   if int(self.m.geom_type[a])==plane or int(self.m.geom_type[b])==plane:planes.append((a,b))
   else:regular.append((a,b))
  def record(a,b,g,axis,motion):
   nonlocal worst,worstv
   robust=g-motion
   row={"geom1":gn(self.m,a),"geom2":gn(self.m,b),"body1":bn(self.m,int(self.m.geom_bodyid[a])),
    "body2":bn(self.m,int(self.m.geom_bodyid[b])),"clearance_lower_bound_mm":g*1000,
    "swept_motion_bound_mm":motion*1000,"swept_clearance_lower_bound_mm":robust*1000}
   if axis is not None:row["separating_axis_world"]=axis
   if robust<worstv:worstv=robust;worst=row
   if robust<CLEARANCE_MM/1000.:
    if len(needs)<6:needs.append(row)
   if details and robust<(CLEARANCE_MM+5.)/1000. and len(rows)<100:rows.append(row)
  for a,b in planes:
   if int(self.m.geom_type[a])==plane:
    n=self.d.geom_xmat[a].reshape(3,3)[:,2];p=self.d.geom_xpos[a]
    local=self.d.geom_xmat[b].reshape(3,3).T@n
    g=float(np.min(self.shape[b].points@local)+self.d.geom_xpos[b]@n-p@n)
   else:
    n=self.d.geom_xmat[b].reshape(3,3)[:,2];p=self.d.geom_xpos[b]
    local=self.d.geom_xmat[a].reshape(3,3).T@n
    g=float(np.min(self.shape[a].points@local)+self.d.geom_xpos[a]@n-p@n)
   motion=(bounds or {}).get(a,0.)+(bounds or {}).get(b,0.)
   record(a,b,g,None,motion)
  if regular:
   aa=np.fromiter((p[0] for p in regular),dtype=np.int32,count=len(regular))
   bb=np.fromiter((p[1] for p in regular),dtype=np.int32,count=len(regular))
   amin=np.asarray([w[g][0] for g in aa]);amax=np.asarray([w[g][1] for g in aa])
   bmin=np.asarray([w[g][0] for g in bb]);bmax=np.asarray([w[g][1] for g in bb])
   dv=np.maximum(0.,np.maximum(amin-bmax,bmin-amax));boxgap=np.linalg.norm(dv,axis=1)
   motion=np.asarray([(bounds or {}).get(int(a),0.)+(bounds or {}).get(int(b),0.) for a,b in zip(aa,bb)])
   far=boxgap>=CLEARANCE_MM/1000.+motion
   for idx in np.flatnonzero(far):
    record(int(aa[idx]),int(bb[idx]),float(boxgap[idx]),None,float(motion[idx]))
   near=np.flatnonzero(~far)
   for idx in near:
    a,b=int(aa[idx]),int(bb[idx]);g,axis=self.pair_gap(a,b,w,con)
    record(a,b,g,axis,float(motion[idx]))
  return {"pass":not needs,"minimum_clearance_mm":None if worst is None else worst["clearance_lower_bound_mm"],
   "minimum_swept_clearance_mm":None if worst is None else worst["swept_clearance_lower_bound_mm"],
   "worst_pair":worst,"needs_subdivision":needs,"near_pairs":rows}

 def pose(self,arm,hand,pairs=None,details=False):
  self.apply(arm,hand);return self.eval_current(arm,hand,pairs or self.path_pairs,None,details)

 def sweep(self,a0,h0,a1,h1,phase,depth=0,nodes=None):
  if nodes is None:nodes={"n":0}
  nodes["n"]+=1
  if nodes["n"]>255:return {"pass":False,"reason":"SWEEP_NODE_BUDGET","phase":phase}
  a0=np.asarray(a0,float);a1=np.asarray(a1,float);am=((a0+a1)/2).tolist();hm={n:(float(h0[n])+float(h1[n]))/2 for n in HAND}
  bounds=self.bounds(a0,h0,a1,h1);self.apply(am,hm);r=self.eval_current(am,hm,self.path_pairs,bounds,False)
  item={"pass":r["pass"],"phase":phase,"depth":depth,"node_count":nodes["n"],
   "minimum_swept_clearance_mm":r["minimum_swept_clearance_mm"],"worst_pair":r["worst_pair"],"q_arm_mid":am}
  if r["pass"]:return item
  if r["worst_pair"] and r["worst_pair"]["clearance_lower_bound_mm"]<=0:
   item["reason"]="COLLISION_OR_UNRESOLVED_OVERLAP";return item
  if depth>=MAX_SWEEP_DEPTH:item["reason"]="SWEPT_CLEARANCE_UNPROVEN_AT_MAX_DEPTH";item["needs_subdivision"]=r["needs_subdivision"];return item
  left=self.sweep(a0,h0,am,hm,phase,depth+1,nodes)
  if not left.get("pass"):return left
  right=self.sweep(am,hm,a1,h1,phase,depth+1,nodes)
  if not right.get("pass"):return right
  return {"pass":True,"phase":phase,"depth":depth,"node_count":nodes["n"],
   "minimum_swept_clearance_mm":min(left["minimum_swept_clearance_mm"],right["minimum_swept_clearance_mm"])}

 def path_candidate(self,c,home,open_hand,ref_pose,limits):
  rot=Rotation.from_euler("z",math.radians(c.get("yaw_deg",0))).as_matrix()
  if c["route"]=="OBLIQUE":rot=Rotation.from_rotvec([0,math.radians(c["tilt_deg"]),0]).as_matrix()
  offset=np.asarray(c.get("offset_xy_m",(0.0,0.0)),dtype=float);pg=np.asarray(ref_pose["position"],float)+[offset[0],offset[1],c["height_offset_m"]];Rg=rot@np.asarray(ref_pose["rotation"],float)
  if c["route"]=="TOP_DOWN":direction=np.array([0.,0.,1.]);standoff=.240
  elif c["route"]=="SIDE_ENTRY":direction=np.array([0.,1.,0.]);standoff=.040
  else:direction=rot@np.array([0.,0.,1.]);direction/=np.linalg.norm(direction);standoff=.220
  pp=pg+direction*standoff;qseed=list(map(float,c.get("ik_seed_arm",home)));qpath=[];ikrows=[]
  baseid=int(mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link"))
  for k,alpha in enumerate(np.linspace(0,1,11)):
   p=pp+(pg-pp)*alpha
   q,ik=bounded.dls_pose(self.m,self.d,qseed,open_hand,self.oq,self.aq,self.obj,self.ob,self.friction,
       baseid,pg,p,Rg,self.ad,limits)
   ikrows.append({"index":k,"alpha":float(alpha),"target_pos_m":p.tolist(),"ik":ik,"q_arm":q})
   if q is None:return {"ik_status":"BLOCKED_IK","failure":ik,"ik_timeline":ikrows,"path_status":"NOT_RUN"}
   qseed=q;qpath.append(q)
  first=self.pose(home,open_hand,details=True);mins=[first["minimum_clearance_mm"]]
  endpoints=[{"name":"HOME","pass":first["pass"],"clearance_mm":first["minimum_clearance_mm"],"worst":first["worst_pair"]}]
  sws=[];ok=first["pass"];block=None if ok else {"phase":"HOME","result":first};prev=home
  for k,q in enumerate(qpath):
   if not ok:break
   ex=self.pose(q,open_hand,details=True);mins.append(ex["minimum_clearance_mm"])
   endpoints.append({"name":f"APPROACH_{k}","pass":ex["pass"],"clearance_mm":ex["minimum_clearance_mm"],"worst":ex["worst_pair"]})
   if not ex["pass"]:ok=False;block={"phase":f"APPROACH_{k}","result":ex};break
   sw=self.sweep(prev,open_hand,q,open_hand,f"APPROACH_{k}");sws.append(sw)
   if not sw.get("pass"):ok=False;block={"phase":f"APPROACH_{k}","sweep":sw};break
   mins.append(sw.get("minimum_swept_clearance_mm",float("inf")));prev=q
  end=self.pose(qpath[-1],open_hand,details=True)
  lim=all(limits[n][0]+.05<=v<=limits[n][1]-.05 for q in qpath for n,v in zip(ARM,q))
  return {"ik_status":"PASS_STATIC_IK" if all(x["ik"]["converged"] for x in ikrows) else "BLOCKED_IK_RESIDUAL",
   "ik_timeline":ikrows,"pregrasp_pos_world_m":pp.tolist(),"target_pos_world_m":pg.tolist(),
   "target_R_world":Rg.tolist(),"approach_direction":direction.tolist(),"standoff_m":standoff,"arm_limits_pass":lim,
   "path_status":"PASS_STATIC_SWEPT" if ok and lim else "BLOCKED_STATIC_SWEPT",
   "minimum_path_clearance_mm":None if not mins else float(min(mins)),"first_path_blocker":block,
   "endpoint_open_clear":end["pass"],"endpoint_open_clearance_mm":end["minimum_clearance_mm"],
   "endpoint_worst_pair":end["worst_pair"],"path_endpoints":endpoints,"swept_segments":sws,
   "q_pregrasp_arm":qpath[0],"q_approach_arm":qpath[-1]}

def clip_poly(poly,z,above):
 if not poly:return []
 out=[];prev=poly[-1];pin=prev[2]>=z if above else prev[2]<=z
 for cur in poly:
  cin=cur[2]>=z if above else cur[2]<=z
  if cin!=pin:
   den=cur[2]-prev[2]
   if abs(den)>1e-18:out.append(prev+((z-prev[2])/den)*(cur-prev))
  if cin:out.append(cur)
  prev,pin=cur,cin
 return out

def exact_source_cylinder(tris,pos,rot,radius,half):
 local=(tris-pos)@rot;lo=local.min(1);hi=local.max(1)
 mask=(lo[:,2]<=half)&(hi[:,2]>=-half)&(lo[:,0]<=radius)&(hi[:,0]>=-radius)&(lo[:,1]<=radius)&(hi[:,1]>=-radius)
 ix=np.flatnonzero(mask);best=math.inf;bestid=None
 for ti in ix:
  poly=clip_poly([local[ti,0],local[ti,1],local[ti,2]],-half,True);poly=clip_poly(poly,half,False)
  if not poly:continue
  signs=[]
  if len(poly)>=3:
   for k in range(len(poly)):
    a=poly[k][:2];b=poly[(k+1)%len(poly)][:2];signs.append((b[0]-a[0])*(-a[1])-(b[1]-a[1])*(-a[0]))
  inside=bool(signs) and (all(x>=-1e-12 for x in signs) or all(x<=1e-12 for x in signs))
  if inside:r=0.
  elif len(poly)==1:r=float(np.linalg.norm(poly[0][:2]))
  else:
   rs=[]
   for k in range(len(poly)):
    a=poly[k];b=poly[(k+1)%len(poly)];ab=b[:2]-a[:2];den=float(ab@ab)
    t=0. if den<1e-24 else float(np.clip((-a[:2]@ab)/den,0,1));rs.append(float(np.linalg.norm(a[:2]+t*ab)))
   r=min(rs)
  if r<best:best=r;bestid=int(ti)
 return {"intersects_cylinder_volume":bool(best<radius-1e-9),"min_axis_radius_mm":None if not math.isfinite(best) else best*1000,
  "radial_intrusion_mm":None if not math.isfinite(best) else max(0.,(radius-best)*1000),"candidate_triangles":int(len(ix)),"triangle":bestid}

def source_visual_state(e,paths,integrity):
 m,d=e.m,e.d;obj=e.obj;pos=d.geom_xpos[obj].astype(float);rot=d.geom_xmat[obj].reshape(3,3).astype(float)
 expected={Path(k).name:v for k,v in integrity["STL_sha256"].items()};out=[]
 for body,fn in SOURCE_BODIES.items():
  b=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body))
  gs=[g for g in range(m.ngeom) if int(m.geom_bodyid[g])==b and gn(m,g).endswith("__visual")]
  if len(gs)!=1:raise ValueError("visual source geom missing/duplicated: "+body)
  g=gs[0];mid=int(m.geom_dataid[g]);p=paths[body];raw=p.read_bytes();n=struct.unpack_from("<I",raw,80)[0]
  if sha(p)!=expected[fn] or n!=int(m.mesh_facenum[mid]):raise ValueError("source STL hash/face count mismatch: "+fn)
  va,nv=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid]);fa,nf=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
  v=m.mesh_vert[va:va+nv].astype(float);f=m.mesh_face[fa:fa+nf].astype(int)
  R=d.geom_xmat[g].reshape(3,3).astype(float);gp=d.geom_xpos[g].astype(float);tris=v[f]@R.T+gp
  out.append({"body":body,"stl":str(p),"sha256":sha(p),"triangles":n,
   "source_surface":exact_source_cylinder(tris,pos,rot,float(m.geom_size[obj,0]),float(m.geom_size[obj,1]))})
 return out

def group_gap(e,arm,hand,bodyset):
 e.apply(arm,hand);gs=[g for g in e.gripper if bn(e.m,int(e.m.geom_bodyid[g])) in bodyset]
 ps=[tuple(sorted((g,e.obj))) for g in gs if base.coll(e.m,g,e.obj,e.excl)]
 metric=e.eval_current(arm,hand,ps,None,True);contacts=[];gb={int(e.m.geom_bodyid[g]) for g in gs}
 for c in e.d.contact[:e.d.ncon]:
  a,b=int(c.geom1),int(c.geom2)
  if e.obj not in (a,b):continue
  other=b if a==e.obj else a
  if int(e.m.geom_bodyid[other]) not in gb:continue
  n=np.asarray(c.frame[:3],float)
  if a!=e.obj:n=-n
  contacts.append({"geom":gn(e.m,other),"body":bn(e.m,int(e.m.geom_bodyid[other])),"distance_mm":float(c.dist)*1000,
   "point_world_m":np.asarray(c.pos).tolist(),
   "point_object_local_m":((np.asarray(c.pos)-e.d.geom_xpos[e.obj])@e.d.geom_xmat[e.obj].reshape(3,3)).tolist(),
   "normal_outward_object_world":n.tolist(),"abs_normal_z":abs(float(n[2]))})
 return metric,contacts

def candidate_defs():
 out=[]
 for route in ("TOP_DOWN","SIDE_ENTRY"):
  for dz in (-.02,0.,.02):
   for yaw in (-15.,0.,15.):
    out.append({"id":f"{route}_Z{int(round(dz*1000)):+d}_YAW{int(yaw):+d}","route":route,"height_offset_m":dz,"yaw_deg":yaw,"tilt_deg":0.})
 for dz in (-.02,0.,.02):
  for tilt in (-20.,20.):
   out.append({"id":f"OBLIQUE_Z{int(round(dz*1000)):+d}_TILT{int(tilt):+d}","route":"OBLIQUE","height_offset_m":dz,"yaw_deg":0.,"tilt_deg":tilt})
 if len(out)!=24:raise AssertionError("24 candidate definition error")
 return out

def main():
 self_test()
 ap=argparse.ArgumentParser();ap.add_argument("--output-parent",default=str(REV/"issue-46"));args=ap.parse_args()
 parent=pg.validate_output_path(args.output_parent,authorized_roots=(str(REV),),require_exists=True,expected_kind="directory",environment="wsl")
 oldf=mujoco.mj_forward;oldsteps=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2);fw=0;steps=0
 def forward(*a,**k):
  nonlocal fw;fw+=1;return oldf(*a,**k)
 def no_step(*a,**k):
  nonlocal steps;steps+=1;raise RuntimeError("physics stepping prohibited")
 mujoco.mj_forward=forward;mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=no_step
 try:
  if base.git(base.WT,"rev-parse","HEAD")!=PR90_HEAD or base.git(base.WT,"status","--porcelain"):raise RuntimeError("PR90 not exact clean read-only input")
  if sha(base.NX)!=PR90_MODEL_SHA or sha(base.OX)!=OLD_MODEL_SHA:raise RuntimeError("pinned input hash mismatch")
  ri=json.loads(base.NR.read_text());profile=base.profile();oldres=json.loads(base.OR.read_text())
  for directory in (base.OLD,base.NEW):
   mf=json.loads((directory/"artifact_paths.json").read_text());pg.verify_artifacts(mf["artifacts"],authorized_roots=(str(REV),),environment="wsl")
  integrity=ri["frozen_candidate_b"]["source_integrity_record"]
  urdf=Path(base.pr90.p.UPSTREAM)/"X2-Ultra_omnipicker.urdf"
  if sha(urdf)!=integrity["URDF_sha256"]:raise RuntimeError("official URDF hash mismatch")
  paths={b:Path(next(p for p in integrity["STL_sha256"] if Path(p).name==fn)) for b,fn in SOURCE_BODIES.items()}
  for b,p in paths.items():
   exp=next(v for k,v in integrity["STL_sha256"].items() if Path(k).name==SOURCE_BODIES[b])
   if not p.is_file() or sha(p)!=exp:raise RuntimeError("vendor STL missing/changed: "+str(p))
  xml=base.make_overlay_xml(base.OX.read_text(),ri);root=ET.fromstring(xml);oldroot=ET.fromstring(base.OX.read_text())
  if base.excl(root)!=base.excl(oldroot):raise RuntimeError("collision exclusions changed")
  m=mujoco.MjModel.from_xml_string(xml);om=mujoco.MjModel.from_xml_path(str(base.OX))
  counts={k:int(getattr(m,k)) for k in ("nbody","njnt","ngeom","nmesh","neq","nu","nq","nv")}
  if counts!={k:int(getattr(om,k)) for k in counts}:raise RuntimeError("topology changed")
  for n in ("geom_contype","geom_conaffinity","geom_bodyid","geom_dataid","geom_pos","geom_quat","body_pos","body_quat","jnt_range","jnt_axis","jnt_pos","actuator_trnid"):
   if not np.array_equal(getattr(om,n),getattr(m,n)):raise RuntimeError("overlay changed "+n)
  d=mujoco.MjData(m);obj=gid(m,base.OBJ);ob=int(m.geom_bodyid[obj]);oq=int(m.jnt_qposadr[jid(m,base.OJ)])
  inertia=[base.MASS*(3*(base.D/2)**2+base.H**2)/12]*2+[.5*base.MASS*(base.D/2)**2]
  if not np.allclose(m.geom_size[obj,:2],[base.D/2,base.H/2],atol=1e-12):raise RuntimeError("bottle dimensions differ; refusing to modify")
  if abs(m.body_mass[ob]-base.MASS)>1e-12 or not np.allclose(m.body_inertia[ob],inertia,atol=1e-12):raise RuntimeError("bottle mass/inertia differ; refusing to modify")
  fr=m.geom_friction[obj].copy();aq=tuple(int(m.jnt_qposadr[jid(m,n)]) for n in ARM);ad=tuple(int(m.jnt_dofadr[jid(m,n)]) for n in ARM)
  limits={n:tuple(map(float,m.jnt_range[jid(m,n)])) for n in ARM};openq=profile[-1]["q"]
  side=json.loads((REV/"issue-46/candidate-b-full-x2-midbody-side-grasp-20261023T063306Z/plan.json").read_text())
  home=list(map(float,side["approach_q"][0]))
  cfg=next(x for x in oldres["candidate_configurations"] if x["id"]=="D70_H230_M550G_MID")
  with base.OC.open(newline="",encoding="utf-8") as f:rows=[x for x in csv.DictReader(f) if x["config"]==cfg["id"]]
  rows.sort(key=lambda x:int(x["sample"]));top_goal=list(map(float,json.loads(rows[-1]["q_arm_json"])));side_goal=list(map(float,side["q_grasp"]))
  baseid=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link"))
  def getpose(q):
   base.state(m,d,q,openq,oq,aq,obj,ob,fr)
   return {"position":d.xpos[baseid].astype(float).tolist(),"rotation":d.xmat[baseid].reshape(3,3).astype(float).tolist(),
       "quat_wxyz":d.xquat[baseid].astype(float).tolist()}
  tp=getpose(top_goal);sp=getpose(side_goal);ev=Evaluator(m,d,root,obj,ob,oq,aq,ad,limits,fr)
  start=ev.pose(home,openq,details=True)
  if not start["pass"]:raise RuntimeError("start state lacks collision clearance: "+str(start["worst_pair"]))
  candidates=[]
  for c in candidate_defs():
   r=ev.path_candidate(c,home,openq,sp if c["route"]=="SIDE_ENTRY" else tp,limits)
   candidates.append({"candidate":c,**r})
   print("COARSE",c["id"],r.get("ik_status"),r.get("path_status"),r.get("minimum_path_clearance_mm"),flush=True)
  eligible=[]
  for c in candidates:
   if c.get("ik_status")!="PASS_STATIC_IK" or c.get("q_approach_arm") is None:continue
   end=ev.pose(c["q_approach_arm"],openq,details=True)
   c["endpoint_open_clear"]=end["pass"];c["endpoint_open_clearance_mm"]=end["minimum_clearance_mm"]
   c["endpoint_worst_pair"]=end["worst_pair"];c["endpoint_near_pairs"]=end["near_pairs"];eligible.append(c)
  eligible.sort(key=lambda c:(c.get("path_status")=="PASS_STATIC_SWEPT",bool(c.get("endpoint_open_clear")),
   float(c.get("minimum_path_clearance_mm") or -1e9),float(c.get("endpoint_open_clearance_mm") or -1e9)),reverse=True)
  refined=eligible[:MAX_REFINED];closures=[];source_out=[]
  for cand in refined:
   arm=cand["q_approach_arm"];cid=cand["candidate"]["id"];ev.apply(arm,openq)
   srcopen=source_visual_state(ev,paths,integrity);source_out.append({"candidate":cid,"state":"OPEN_TARGET","meshes":srcopen})
   sequence=list(reversed(profile));first_other=None;first_self=None;first_n=None;first_w=None;first_bi=None;timeline=[]
   nonpad=[]
   for p in ev.gr_obj:
    other=p[1] if p[0]==obj else p[0]
    if bn(m,int(m.geom_bodyid[other])) not in PAD_BODIES:nonpad.append(p)
   for step,e in enumerate(sequence):
    hand=e["q"];ev.apply(arm,hand)
    ng,nc=group_gap(ev,arm,hand,{"R_hand_narrow3_Link"});wg,wc=group_gap(ev,arm,hand,{"R_hand_wide3_Link"})
    og=ev.eval_current(arm,hand,nonpad,None,True);touch=lambda cs:[x for x in cs if x["distance_mm"]<=CONTACT_TOL_MM]
    nt,wt=touch(nc),touch(wc)
    sc=[]
    for contact in ev.d.contact[:ev.d.ncon]:
     a,b=int(contact.geom1),int(contact.geom2)
     if ev.obj in (a,b):continue
     ba,bb=bn(m,int(m.geom_bodyid[a])),bn(m,int(m.geom_bodyid[b]))
     if ba.startswith(("R_hand_","R_omnipicker")) and bb.startswith(("R_hand_","R_omnipicker")) and float(contact.dist)<=CONTACT_TOL_MM/1000.:
      sc.append({"geom1":gn(m,a),"geom2":gn(m,b),"body1":ba,"body2":bb,"distance_mm":float(contact.dist)*1000})
    if nt and first_n is None:first_n={"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"contacts":nt,"gap_mm":ng["minimum_clearance_mm"]}
    if wt and first_w is None:first_w={"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"contacts":wt,"gap_mm":wg["minimum_clearance_mm"]}
    if og["minimum_clearance_mm"] is not None and og["minimum_clearance_mm"]<=CONTACT_TOL_MM and first_other is None:
     first_other={"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"clearance_mm":og["minimum_clearance_mm"],"pair":og["worst_pair"]}
    if sc and first_self is None:first_self={"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"pairs":sc[:8]}
    inband=lambda cs:bool(cs) and all(abs(float(x["point_object_local_m"][2])*1000-cand["candidate"]["height_offset_m"]*1000)<=CONTACT_BAND_HALF_MM for x in cs)
    if inband(nt) and inband(wt) and first_bi is None:
     nn=np.mean([x["normal_outward_object_world"] for x in nt],axis=0);wn=np.mean([x["normal_outward_object_world"] for x in wt],axis=0)
     if np.linalg.norm(nn):nn/=np.linalg.norm(nn)
     if np.linalg.norm(wn):wn/=np.linalg.norm(wn)
     first_bi={"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"contacts":nt+wt,
       "pad_gaps_mm":{"narrow":ng["minimum_clearance_mm"],"wide":wg["minimum_clearance_mm"],"nonpad":og["minimum_clearance_mm"]},
       "normal_dot":float(np.dot(nn,wn)),"max_abs_normal_z":float(max(abs(nn[2]),abs(wn[2]))),
       "friction":m.geom_friction[obj].tolist(),
       "pad_height_difference_mm":abs(float(np.mean([x["point_object_local_m"][2] for x in nt]))-float(np.mean([x["point_object_local_m"][2] for x in wt])))*1000}
     break
    timeline.append({"step":step,"percent":e["percent"],"qCAD":e["qCAD"],"arm_q_rad":arm,
       "pad_clearance_mm":{"narrow":ng["minimum_clearance_mm"],"wide":wg["minimum_clearance_mm"]},
       "nonpad_clearance_mm":og["minimum_clearance_mm"],"nonpad_pair":og["worst_pair"],
       "pad_touch_contacts":{"narrow":nt,"wide":wt},"internal_contacts":sc[:8],
       "hand_limits_pass":all(not m.jnt_limited[jid(m,n)] or m.jnt_range[jid(m,n),0]<=v<=m.jnt_range[jid(m,n),1] for n,v in hand.items())})
    if first_other and first_bi is None:break
   topology=bool(first_bi and (first_other is None or first_bi["step"]<first_other["step"]) and first_self is None)
   srcc=None
   if first_bi:
    ev.apply(arm,sequence[first_bi["step"]]["q"]);srcc=source_visual_state(ev,paths,integrity)
    source_out.append({"candidate":cid,"state":"FIRST_BILATERAL","meshes":srcc})
   closures.append({"candidate_id":cid,"path_status":cand.get("path_status"),
    "first_narrow_pad":first_n,"first_wide_pad":first_w,"first_nonpad_contact":first_other,
    "first_internal_contact":first_self,"first_bilateral_in_band":first_bi,"pad_first":topology,
    "closure_status":"PASS_STATIC_PAD_FIRST" if topology else "BLOCKED_STATIC_CLOSURE",
    "timeline":timeline,"source_open":srcopen,"source_at_bilateral":srcc})
  if steps!=0:raise RuntimeError("mj_step was called")
  if fw==0:raise RuntimeError("no mj_forward calls recorded")
  out=pg.create_unique_output_directory(parent.wsl_path,prefix="pr90-full-x2-bounded-24-static",authorized_roots=(str(REV),),environment="wsl")
  model_file=out.native_path/"full_x2_pr90_bounded_static.xml"
  pg.write_new_text(model_file,xml+"\n",authorized_roots=(str(REV),),environment="wsl")
  result={"schema":"robotsim.issue46.pr90_full_x2_bounded_static_grasp.v1","run_id":str(uuid.uuid4()),
   "classification":"SIMULATION_ONLY_STATIC_GEOMETRY_AND_IK","mujoco":mujoco.__version__,"physics_steps":steps,"mj_forward_calls":fw,
   "source_branch":base.git(ROOT,"branch","--show-current"),"source_head":base.git(ROOT,"rev-parse","HEAD"),
   "source_worktree_status":base.git(ROOT,"status","--porcelain"),
   "PR90_head":base.git(base.WT,"rev-parse","HEAD"),"PR90_model_sha256":sha(base.NX),"PR90_range_sha256":sha(base.NR),
   "PR90_profile_sha256":sha(base.NP),"official_URDF_path":str(urdf),"official_URDF_sha256":sha(urdf),
   "official_URDF_revision":"575cc6b988f976c23550e0db85aa1e5475d3652d",
   "official_STLs":[{"path":str(p),"sha256":sha(p)} for p in paths.values()],
   "old_full_x2_model_sha256":sha(base.OX),"generated_full_x2_model_sha256":sha(model_file),
   "wrist_mount":base.body_mount(m),"side_entry_pregrasp_reference":{"plan_path":str(REV/"issue-46/candidate-b-full-x2-midbody-side-grasp-20261023T063306Z/plan.json"),"pregrasp_point_world_m":side["pregrasp_point"],"grasp_point_world_m":side["grasp_point"],"delta_m":(np.asarray(side["pregrasp_point"],float)-np.asarray(side["grasp_point"],float)).tolist()},"model_counts":counts,
   "object":{"type":"upright_free_cylinder","diameter_m":base.D,"height_m":base.H,"mass_kg":base.MASS,
    "center_world_m":list(base.CENTER),"friction":m.geom_friction[obj].tolist(),
    "table_top_z_m":float(d.geom_xpos[base.gid(m,base.TABLE)][2]+m.geom_size[base.gid(m,base.TABLE),2])},
   "collision_policy":{"active_PR90_geoms":len(ev.gripper),"original_exclusions":len(ev.excl),"clearance_mm":CLEARANCE_MM,
    "geometry_method":"compiled collision mesh vertices with sampled source-face projection witnesses; positive gaps are conservative and missing witnesses unresolved; exact boxes and circumscribed 96-sided cylinder; no mj_geomDistance",
    "swept_method":"recursive joint-space arc-length bound from revolute axes through rigid body-frame offsets; per-segment node cap 255 and depth 7",
    "allowed_support":"ankle roll links against floor only"},
   "candidate_count":len(candidates),"refined_count":len(refined),"start_state":{"arm_q_rad":home,"clearance_mm":start["minimum_clearance_mm"],"worst_pair":start["worst_pair"]},
   "candidates":candidates,"refined_closures":closures,"source_surface_checks":source_out,
   "candidate_set":candidate_defs(),"stages":{"PREGRASP_IK":"STATIC_ONLY","APPROACH":"STATIC_ONLY","GRASP":"NOT RUN",
    "LOAD_TRANSFER":"NOT RUN","LIFT_50MM":"NOT RUN","TRANSFER":"NOT RUN","RELEASE":"NOT RUN","PLACE":"NOT RUN","SETTLE":"NOT RUN"}}
  result["output"]=out.as_dict();result["runner_path"]=str(Path(__file__).resolve());result["runner_sha256"]=sha(Path(__file__).resolve())
  pg.write_new_text(out.native_path/"result.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",authorized_roots=(str(REV),),environment="wsl")
  cols=["candidate","route","height_offset_m","yaw_deg","tilt_deg","ik_status","arm_limits_pass","path_status",
    "minimum_path_clearance_mm","endpoint_open_clear","endpoint_open_clearance_mm","first_path_blocker"]
  st=io.StringIO();w=csv.DictWriter(st,fieldnames=cols,extrasaction="ignore");w.writeheader()
  for c in candidates:w.writerow({k:json.dumps(c.get(k),ensure_ascii=False) if isinstance(c.get(k),(dict,list)) else c.get(k,"") for k in cols})
  pg.write_new_text(out.native_path/"candidate_summary.csv",st.getvalue(),authorized_roots=(str(REV),),environment="wsl")
  cols=["candidate_id","path_status","closure_status","first_narrow_pad","first_wide_pad","first_nonpad_contact","first_internal_contact","first_bilateral_in_band","pad_first"]
  st=io.StringIO();w=csv.DictWriter(st,fieldnames=cols,extrasaction="ignore");w.writeheader()
  for c in closures:w.writerow({k:json.dumps(c.get(k),ensure_ascii=False) if isinstance(c.get(k),(dict,list)) else c.get(k,"") for k in cols})
  pg.write_new_text(out.native_path/"closure_summary.csv",st.getvalue(),authorized_roots=(str(REV),),environment="wsl")
  lines=["# PR90 full-X2 bounded static grasp screen","",f"Run ID: {result['run_id']}",f"WSL: {out.wsl_path}",f"UNC: {out.windows_unc}","",
   f"MuJoCo {mujoco.__version__}; mj_step calls {steps}; mj_forward calls {fw}. Static-only, no grasp run.",
   f"PR90 {PR90_HEAD}; official X2 pin 575cc6b988f976c23550e0db85aa1e5475d3652d. Bottle retained at D={base.D}m H={base.H}m mass={base.MASS}kg.",
   f"Active PR90 collision geoms {len(ev.gripper)}; original exclusions {len(ev.excl)}; clearance threshold {CLEARANCE_MM} mm.",
   "Swept paths use recursive joint-space arc bounds. Mesh distances use compiled convex hulls and SAT; no mj_geomDistance zero values are used.","","## Coarse 24"]
  for c in candidates:lines.append(f"- {c['candidate']['id']}: IK={c.get('ik_status')}; path={c.get('path_status')}; min={c.get('minimum_path_clearance_mm')}; open endpoint={c.get('endpoint_open_clear')}; blocker={c.get('first_path_blocker')}.")
  lines.extend(["","## Refined closure"])
  for c in closures:lines.append(f"- {c['candidate_id']}: {c['closure_status']}; narrow={c['first_narrow_pad']}; wide={c['first_wide_pad']}; nonpad={c['first_nonpad_contact']}; bilateral={c['first_bilateral_in_band']}.")
  lines.extend(["","No force transfer or lift is claimed. Physical stages remain NOT RUN."])
  pg.write_new_text(out.native_path/"REPORT.md","\n".join(lines)+"\n",authorized_roots=(str(REV),),environment="wsl")
  cmd=f"cd {ROOT}\nenv PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python {Path(__file__).resolve()} --output-parent {parent.wsl_path}\n"
  pg.write_new_text(out.native_path/"REPRODUCE.md",cmd,authorized_roots=(str(REV),),environment="wsl")
  manifest=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
  pg.verify_artifacts(manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
  print(json.dumps({"run_id":result["run_id"],"output":out.as_dict(),"steps":steps,"forwards":fw,
   "candidate_count":len(candidates),"refined_count":len(closures),
   "candidates":[{"id":c["candidate"]["id"],"ik":c.get("ik_status"),"path":c.get("path_status"),"clearance":c.get("minimum_path_clearance_mm")} for c in candidates],
   "manifest":len(manifest["artifacts"])},indent=2))
  return 0
 finally:
  mujoco.mj_forward=oldf;mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=oldsteps

if __name__=="__main__":raise SystemExit(main())
