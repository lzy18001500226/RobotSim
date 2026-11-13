"""Sensor-contract and fitting regressions, no simulator/ROS rollout."""
import ast
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np
from scripts.research.issue46.perception.native_rgbd_grasp import estimate,qualify_estimate
from scripts.research.issue46.grasp.stl_approved_circle_ik import circle_task_jacobian,axis_grasp_error
from scripts.research.issue46.native_runtime.test_pregrasp_feedback import generated

class GravityComponentTest(unittest.TestCase):
    def test_foreshortened_bottle_and_blue_floor_are_separated_by_depth(self):
        from scripts.research.issue46.perception.native_rgbd_grasp import vertical_component
        mask=np.zeros((50,60),dtype=bool);height=np.zeros(mask.shape)
        mask[3:15,3:13]=True
        height[3:15,3:13]=np.linspace(.55,.77,12)[:,None]
        mask[25:49,1:59]=True
        result=vertical_component(mask,world_height=height)
        self.assertIsNotNone(result)
        self.assertEqual(result[1]["bbox_xyxy"],[3,3,13,15])
        self.assertIsNone(vertical_component(mask))
        height[3:15,3:13]=.6
        self.assertIsNone(vertical_component(mask,world_height=height))
    def test_mixed_boundary_depth_cannot_make_blue_floor_beat_bottle(self):
        from scipy.ndimage import binary_erosion
        from scripts.research.issue46.perception.native_rgbd_grasp import vertical_component
        mask=np.zeros((70,90),dtype=bool);height=np.zeros(mask.shape)
        mask[3:25,3:14]=True
        height[3:25,3:14]=np.linspace(.55,.77,22)[:,None]
        floor=np.zeros_like(mask);floor[35:69,1:89]=True
        mask|=floor;height[floor]=.002
        height[floor&~binary_erosion(floor)]=.25
        result=vertical_component(mask,world_height=height)
        self.assertEqual(result[1]['bbox_xyxy'],[3,3,14,25])

    def test_nonfinite_height_cannot_qualify(self):
        from scripts.research.issue46.perception.native_rgbd_grasp import vertical_component
        mask=np.ones((12,10),dtype=bool);height=np.linspace(0,.2,120).reshape(mask.shape);height[0,0]=np.nan
        self.assertIsNone(vertical_component(mask,world_height=height))

def synthetic_cylinder(radius):
    w,h=320,240;f=210.;K={'fx':f,'fy':f,'cx':(w-1)/2,'cy':(h-1)/2}
    a=.65;T=np.eye(4);T[:3,:3]=np.array([[0.,-math.sin(a),math.cos(a)],[-1.,0.,0.],[0.,-math.cos(a),-math.sin(a)]]);T[:3,3]=[0,0,.95]
    y,x=np.indices((h,w));rays=np.stack(((x-K['cx'])/f,(y-K['cy'])/f,np.ones_like(x)),axis=-1)@T[:3,:3].T
    origin=T[:3,3];center=np.array([.5,.015,.7]);delta=origin[:2]-center[:2]
    A=np.sum(rays[:,:,:2]**2,axis=-1);B=2*np.sum(rays[:,:,:2]*delta,axis=-1);C=np.sum(delta**2)-radius**2;D=B*B-4*A*C
    side=np.where(D>=0,(-B-np.sqrt(np.maximum(0,D)))/(2*A),np.inf);zs=origin[2]+side*rays[:,:,2]
    side=np.where((side>0)&(zs>=.55)&(zs<=.85),side,np.inf)
    cap=(.85-origin[2])/rays[:,:,2];capxy=origin[:2]+cap[:,:,None]*rays[:,:,:2]
    cap=np.where((cap>0)&(np.linalg.norm(capxy-center[:2],axis=-1)<=radius),cap,np.inf)
    depth=np.minimum(side,cap);rgb=np.zeros((h,w,3),dtype=np.uint8);rgb[np.isfinite(depth)]=[20,70,220]
    return rgb,depth,K,T,center

class SensorTests(unittest.TestCase):
    def test_snapshot_requires_fresh_independent_reobservation_and_expires(self):
        import copy
        from scripts.research.issue46.perception.native_rgbd_grasp import make_stationary_snapshot,qualify_stationary_snapshot
        first={'pass':True,'generation':1,'source_host_ns':1000000000,'created_host_ns':1050000000,'estimated_center_world_m':[.5,0.,.7],'axis_world':[0.,0.,1.],'diameter_m':.07,'object_truth_used_by_estimator':False}
        second={**first,'source_host_ns':2000000000,'created_host_ns':2050000000}
        snapshot=make_stationary_snapshot(first,second,generation=1,now_ns=2100000000)
        self.assertEqual(qualify_stationary_snapshot(snapshot,generation=1,now_ns=3000000000)['position_m'],first['estimated_center_world_m'])
        self.assertEqual(snapshot['original_observation']['source_host_ns'],1000000000)
        with self.assertRaisesRegex(RuntimeError,'DEADLINE'):qualify_stationary_snapshot(snapshot,generation=1,now_ns=48000000000)
        with self.assertRaisesRegex(RuntimeError,'CONTRACT'):qualify_stationary_snapshot(snapshot,generation=2,now_ns=3000000000)
        changed=copy.deepcopy(snapshot);changed['estimated_center_world_m'][0]+=.001
        with self.assertRaisesRegex(RuntimeError,'CHANGED'):qualify_stationary_snapshot(changed,generation=1,now_ns=3000000000)
        with self.assertRaisesRegex(RuntimeError,'INDEPENDENT'):make_stationary_snapshot(first,first,generation=1,now_ns=1100000000)
        second['estimated_center_world_m']=[.51,0.,.7]
        with self.assertRaisesRegex(RuntimeError,'NOT_STATIONARY'):make_stationary_snapshot(first,second,generation=1,now_ns=2100000000)

    def test_only_initial_readiness_can_await_a_new_image_without_refreshing_old_stamp(self):
        import json
        from types import SimpleNamespace as NS
        from scripts.research.issue46.perception import native_rgbd_grasp as sensor
        clock=NS(now=2.)
        record={'pass':True,'generation':1,'source_host_ns':1000000000,'estimated_center_world_m':[.5,0.,.7],'axis_world':[0.,0.,1.],'diameter_m':.07,'object_truth_used_by_estimator':False}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'latest.json';p.write_text(json.dumps(record))
            def new_frame(dt):
                clock.now+=dt;record['source_host_ns']=int(clock.now*1e9);p.write_text(json.dumps(record))
            with mock.patch.object(sensor.time,'monotonic',side_effect=lambda:clock.now),mock.patch.object(sensor.time,'monotonic_ns',side_effect=lambda:int(clock.now*1e9)),mock.patch.object(sensor.time,'sleep',side_effect=new_frame) as sleep:
                with self.assertRaisesRegex(RuntimeError,'STALE'):sensor.qualify_estimate(record,generation=1)
                observed,_=sensor.await_initial_estimate(p,generation=1)
                self.assertGreater(observed['source_host_ns'],1000000000);sleep.assert_called_once()
                record['pass']=False;p.write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError,'NOT_QUALIFIED'):sensor.await_initial_estimate(p,generation=1)

    def test_compiled_cache_requires_exact_source_engine_and_binary(self):
        import json,hashlib
        from scripts.research.issue46.perception import native_rgbd_grasp as sensor
        def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scene=root/'scene.xml';scene.write_text('<mujoco/>');binary=root/'m.mjb';binary.write_bytes(b'fixture')
            record={'source_sha256':digest(scene),'engine':mujoco.mj_version(),'sensor_source_sha256':digest(Path(sensor.__file__)),'binary':str(binary),'binary_sha256':digest(binary)}
            p=root/'cache.json';p.write_text(json.dumps(record))
            with mock.patch.object(sensor.mujoco.MjModel,'from_binary_path',return_value='verified'):
                self.assertEqual(sensor.load_cached_model(scene,p),'verified')
                binary.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'mismatch'):sensor.load_cached_model(scene,p)

    def test_renderer_warmup_cannot_be_published_as_fresh_feedback(self):
        import json
        from types import SimpleNamespace as NS
        from scripts.research.issue46.perception import native_rgbd_grasp as sensor
        clock=NS(now=1.);calls=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'sensor';scene=root/'scene.xml';scene.write_text('<mujoco/>')
            latest=mock.Mock();latest.read_text.side_effect=lambda:json.dumps({'host_monotonic_ns':int(clock.now*1e9),'generation':1})
            renderer=mock.Mock()
            def render(*args):
                calls.append(args[-1]['host_monotonic_ns']);clock.now+=.30 if len(calls)==1 else .02
                return np.zeros((3,3,3),np.uint8),np.ones((3,3)),{},np.eye(4)
            def infer(*args):
                (output/'stop').write_text('test stop after one qualified frame')
                return {'center_xy_world_m':[.5,0.],'top_world_z_m':.85,'axis_world':[0.,0.,1.],'diameter_m':.07},np.ones((3,3),bool)
            with mock.patch.object(sensor,'render_backend_info',return_value={'renderer':'test-context'}),mock.patch.object(sensor.pg,'write_new_text',side_effect=lambda path,text:path.write_text(text)),mock.patch.object(sensor,'build_sensor',return_value=(None,None,renderer,0)),mock.patch.object(sensor,'render',side_effect=render),mock.patch.object(sensor,'estimate',side_effect=infer),mock.patch.object(sensor.time,'monotonic',side_effect=lambda:clock.now),mock.patch.object(sensor.time,'monotonic_ns',side_effect=lambda:int(clock.now*1e9)),mock.patch.object(sensor.time,'sleep',side_effect=lambda dt:setattr(clock,'now',clock.now+dt)):
                sensor.stream(scene,latest,output,half_height_m=.15)
            record=json.loads((output/'latest.json').read_text())
            self.assertNotEqual(record['source_host_ns'],calls[0]);self.assertEqual(record['source_host_ns'],calls[1]);self.assertTrue(record['pass'])
            self.assertEqual(len(json.loads((output/'trace.json').read_text())),1)
            renderer.close.assert_called_once()

    def test_variable_radius_is_measured_from_depth(self):
        for radius in (.025,.035):
            rgb,depth,K,T,center=synthetic_cylinder(radius)
            result,_=estimate(rgb,depth,K,T)
            np.testing.assert_allclose(result['center_xy_world_m'],center[:2],atol=.0002)
            self.assertAlmostEqual(result['radius_m'],radius,delta=.0002)
            self.assertAlmostEqual(result['top_world_z_m'],.85,delta=.0002)

    def test_foreground_occlusion_can_separate_a_depth_consistent_cap(self):
        rgb,depth,K,T,center=synthetic_cylinder(.035)
        y,x=np.indices(depth.shape)
        valid_depth=np.where(np.isfinite(depth),depth,0.)
        height=(T[2,0]*(x-K['cx'])*valid_depth/K['fx']+
                T[2,1]*(y-K['cy'])*valid_depth/K['fy']+T[2,2]*valid_depth+T[2,3])
        rgb[(height>.74)&(height<.80)]=0
        result,_=estimate(rgb,depth,K,T)
        np.testing.assert_allclose(result['center_xy_world_m'],center[:2],atol=.0003)
        self.assertAlmostEqual(result['top_world_z_m'],.85,delta=.0003)
        self.assertIsNotNone(result['disconnected_cap_association'])

    def test_disconnected_cap_must_match_barrel_in_3d(self):
        from scripts.research.issue46.perception.native_rgbd_grasp import occluded_cap_points
        mask=np.zeros((30,30),dtype=bool);mask[5:20,5:20]=True
        depth=np.ones(mask.shape);K={'fx':100.,'fy':100.,'cx':12.,'cy':12.}
        T=np.eye(4)
        cap,record=occluded_cap_points(mask,np.zeros_like(mask),depth,K,T,
                                       np.array([1.,0.,.035]),.97)
        self.assertIsNone(cap);self.assertIsNone(record)
        # A distant coaxial blue surface must not be joined across arbitrary gaps.
        cap,record=occluded_cap_points(mask,np.zeros_like(mask),depth,K,T,
                                       np.array([0.,0.,.15]),.5)
        self.assertIsNone(cap);self.assertIsNone(record)

    def test_foreground_depth_at_blue_silhouette_cannot_pull_circle_onto_hand(self):
        from scipy.ndimage import binary_erosion
        rgb,depth,K,T,center=synthetic_cylinder(.035);mask=rgb[:,:,2]>0
        boundary=mask&~binary_erosion(mask);depth[boundary]*=.75
        result,_=estimate(rgb,depth,K,T)
        np.testing.assert_allclose(result['center_xy_world_m'],center[:2],atol=.0003)
        self.assertAlmostEqual(result['radius_m'],.035,delta=.0003)
        self.assertGreater(result['excluded_boundary_pixels'],0)

    def test_duplicate_records_cannot_refresh_sensor_source_age(self):
        record={'pass':True,'generation':2,'source_host_ns':1000000000,'estimated_center_world_m':[.5,0,.7],'axis_world':[0.,0.,1.],'diameter_m':.07,'object_truth_used_by_estimator':False}
        self.assertEqual(qualify_estimate(record,generation=2,now_ns=1100000000)['position_m'],[.5,0.,.7])
        record['created_host_ns']=1300000000
        with self.assertRaisesRegex(RuntimeError,'STALE'):qualify_estimate(record,generation=2,now_ns=1300000000)
        with self.assertRaisesRegex(RuntimeError,'GENERATION'):qualify_estimate(record,generation=3,now_ns=1100000000)
        record['object_truth_used_by_estimator']=True
        with self.assertRaisesRegex(RuntimeError,'PROVENANCE'):qualify_estimate(record,generation=2,now_ns=1100000000)

    def test_generated_arrival_uses_estimate_and_rejects_missing_or_stale_data(self):
        code=generated();function=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='native_circle_arrival')
        import io,json
        row={'host_monotonic_ns':1000000000,'generation':1,'sim_time_s':1.,'bodies':{'R_omnipicker_base_link':{'position_m':[0.,0.,0.],'rotation':np.eye(3).reshape(-1).tolist()},'test_cylinder':{'position_m':[100.,0.,0.],'rotation':np.eye(3).reshape(-1).tolist()}}}
        sensor={'pass':True,'generation':1,'source_host_ns':1000000000,'estimated_center_world_m':[0.,0.,0.],'axis_world':[0.,0.,1.],'object_truth_used_by_estimator':False}
        row['bodies']['R_omnipicker_base_link']['rotation']=[0.,1.,0.,0.,0.,1.,1.,0.,0.]
        ns={'math':math,'json':json,'os':mock.Mock(),'time':mock.Mock(monotonic_ns=lambda:1100000000)};ns['os'].path.exists.return_value=False;exec(compile(ast.Module(body=[function],type_ignores=[]),'arrival','exec'),ns)
        contract={'circle_center_local_m':[0.,0.,0.],'height_above_center_m':0.,'perception_latest':'rgbd'}
        def opened(path,*args,**kwargs):return io.BytesIO((json.dumps(row)+'\n').encode()) if path.endswith('native_latest.json') else io.StringIO(json.dumps(sensor))
        with mock.patch('builtins.open',side_effect=opened):
            result=ns['native_circle_arrival'](contract);self.assertTrue(result['pass']);self.assertEqual(result['target_source'],'FRESH_SIMULATED_HEAD_RGBD')
            sensor['source_host_ns']=0
            with self.assertRaisesRegex(RuntimeError,'RGBD_ARRIVAL'):ns['native_circle_arrival'](contract)
        # A previously measured arrival may freeze its image target. This is
        # explicit bounded proprioceptive continuation, not a refreshed image.
        lock={'phase':'ARRIVAL_HOLD','native_arrival_pass':True,'target_source':'RGBD_FROZEN_AT_VERIFIED_ARRIVAL','created_host_ns':1100000000,'image_source_host_ns':1000000000,'generation':1,'bottle_center_world_m':[0.,0.,0.],'bottle_axis_world':[0.,0.,1.]}
        row['host_monotonic_ns']=2000000000;ns['time'].monotonic_ns=lambda:2100000000;ns['os'].path.exists.return_value=True
        def frozen_open(path,*args,**kwargs):return io.BytesIO((json.dumps(row)+'\n').encode()) if path.endswith('native_latest.json') else io.StringIO(json.dumps(lock if path.endswith('native_tracking_phase.json') else sensor))
        with mock.patch('builtins.open',side_effect=frozen_open):
            result=ns['native_circle_arrival'](contract);self.assertTrue(result['pass']);self.assertEqual(result['image_source_host_ns'],1000000000);self.assertEqual(result['target_source'],'RGBD_FROZEN_AT_VERIFIED_ARRIVAL')
            row['host_monotonic_ns']=15000000000;ns['time'].monotonic_ns=lambda:15100000000
            with self.assertRaisesRegex(RuntimeError,'EXPIRED'):ns['native_circle_arrival'](contract)
        from scripts.research.issue46.perception.native_rgbd_grasp import make_stationary_snapshot
        first={'pass':True,'generation':1,'source_host_ns':1000000000,'created_host_ns':1050000000,'estimated_center_world_m':[0.,0.,0.],'axis_world':[0.,0.,1.],'diameter_m':.07,'object_truth_used_by_estimator':False}
        second={**first,'source_host_ns':2000000000,'created_host_ns':2050000000}
        snapshot=make_stationary_snapshot(first,second,generation=1,now_ns=2100000000)
        contract={'circle_center_local_m':[0.,0.,0.],'height_above_center_m':0.,'perception_snapshot':'snapshot'}
        row['host_monotonic_ns']=3000000000;ns['time'].monotonic_ns=lambda:3100000000
        def snapshot_open(path,*args,**kwargs):return io.BytesIO((json.dumps(row)+'\n').encode()) if path.endswith('native_latest.json') else io.StringIO(json.dumps(snapshot))
        with mock.patch('builtins.open',side_effect=snapshot_open):
            result=ns['native_circle_arrival'](contract);self.assertTrue(result['pass']);self.assertEqual(result['target_source'],'RGBD_BOUNDED_STATIONARY_OBJECT_SNAPSHOT')
            self.assertEqual(result['image_source_host_ns'],1000000000)
            row['host_monotonic_ns']=48000000000;ns['time'].monotonic_ns=lambda:48100000000
            with self.assertRaisesRegex(RuntimeError,'EXPIRED'):ns['native_circle_arrival'](contract)

    def test_circle_analytic_jacobian_matches_central_difference(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint axis="0 1 0"/><geom size=".1"/><body pos=".2 0 .1"><joint axis="0 0 1"/><geom size=".02"/></body></body></worldbody></mujoco>')
        data=mujoco.MjData(model);center=np.array([0.,.02,.13]);axis=np.array([1.,0.,0.]);q=np.array([.3,-.4])
        def residual(q):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            return axis_grasp_error(data.xpos[2],data.xmat[2].reshape(3,3),center,np.zeros(3),axis)
        residual(q);J=circle_task_jacobian(model,data,2,center,axis,np.arange(2));numeric=[]
        for k in range(2):
            delta=np.eye(2)[k]*1e-6;numeric.append((residual(q+delta)-residual(q-delta))/(2e-6))
        np.testing.assert_allclose(J,np.array(numeric).T,atol=1e-8)

if __name__=='__main__':unittest.main()
