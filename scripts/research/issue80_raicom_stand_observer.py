#!/usr/bin/env python3
"""One native SD request after verified Reset; no velocity or input-source API."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from aimdk_msgs.msg import CommonState
from aimdk_msgs.srv import GetMcAction, SetMcAction

from issue80_raicom_reset_only_observer import (
    ResetOnlyObserver, _safe, _write_new_json, _write_snapshot_replace,
    MAX_STATE_AGE_S, MIN_ROOT_Z_M, MAX_ABS_TILT_RAD, TOPIC_TYPES,
    classify_reset_effect, _stamp_ns,
)

SET_SERVICE = '/aimdk_5Fmsgs/srv/SetMcAction'
GET_SERVICE = '/aimdk_5Fmsgs/srv/GetMcAction'
STABLE_HOLD_S = 2.5
MAX_PLANAR_SPEED_M_S = .20
JD_READY_TIMEOUT_S = 1.0
SD_READY_TIMEOUT_S = 2.0


def posture_safety_reason(pose):
    try:
        values = [float(pose[k]) for k in ('z_m','roll_rad','pitch_rad')]
        frame_values = [float(pose[k]) for k in ('x_m','y_m','yaw_rad')]
        velocity = [float(v) for v in pose['linear_velocity_m_s']]
        if not all(math.isfinite(v) for v in values+velocity+frame_values):
            return 'NONFINITE_POSTURE_OR_TWIST'
        if values[0] < MIN_ROOT_Z_M:
            return 'ROOT_HEIGHT_BELOW_MINIMUM'
        if max(abs(values[1]),abs(values[2])) >= MAX_ABS_TILT_RAD:
            return 'ROLL_OR_PITCH_LIMIT'
        if math.hypot(*velocity[:2]) > MAX_PLANAR_SPEED_M_S:
            return 'PLANAR_SPEED_LIMIT'
        return None
    except (KeyError,TypeError,ValueError):
        return 'MALFORMED_POSTURE_OR_TWIST'


def safe_posture(pose):
    return posture_safety_reason(pose) is None


def request_allowed(*, reset_verified, reset_effect, fresh, pose, advancing, count):
    return (reset_verified and reset_effect and fresh and safe_posture(pose)
            and advancing and count == 0)


def source_progress(previous, stamp):
    if stamp is None:return 'MISSING'
    if previous is None or stamp>previous:return 'ADVANCE'
    return 'DUPLICATE' if stamp==previous else 'REGRESSION'


def velocity_graph_snapshot(node):
    endpoints=node.get_publishers_info_by_topic('/aima/mc/locomotion/velocity')
    return {'status':'PASS_ZERO_VELOCITY_PUBLISHERS' if len(endpoints)==0 else 'BLOCKED_VELOCITY_PUBLISHER',
            'publisher_count':len(endpoints),'method':'rclpy_existing_observer_graph',
            'nodes':[{'name':e.node_name,'namespace':e.node_namespace} for e in endpoints]}


def active_mode_agrees(common, readback):
    return (readback.get('header',{}).get('code')==0
            and readback.get('info',{}).get('action_desc')=='STAND_DEFAULT'
            and readback.get('info',{}).get('status',{}).get('value')==100
            and common.get('action_info',{}).get('action_desc')=='STAND_DEFAULT'
            and common.get('action_info',{}).get('status',{}).get('value')==100
            and common.get('body_status',{}).get('value')==1
            and common.get('fsm_state',{}).get('current_state')==4)


def native_transition_state(phase, common, readback, *, elapsed_s, jd_accepted,
                            sd_accepted, initial_fsm6=True, left_initial_fsm6=False):
    """Pinned Run65 semantics; pending never grants action or hold permission."""
    ci=common.get('action_info',{})
    gi=readback.get('info',{})
    mode=ci.get('action_desc'); get_mode=gi.get('action_desc')
    status=ci.get('status',{}).get('value'); get_status=gi.get('status',{}).get('value')
    fsm=common.get('fsm_state',{}).get('current_state')
    body=common.get('body_status',{}).get('value')
    if readback.get('header',{}).get('code')!=0:
        return 'PROTECTION_AMBIGUOUS'
    if phase=='JD':
        if fsm==6 and (left_initial_fsm6 or not initial_fsm6):
            return 'PROTECTION_CONFIRMED'
        if mode not in ('PASSIVE_DEFAULT','JOINT_DEFAULT') or get_mode not in ('PASSIVE_DEFAULT','JOINT_DEFAULT'):
            return 'PROTECTION_AMBIGUOUS'
        if fsm not in (1,6) or body not in (0,1) or status not in (100,200,300) or get_status not in (100,200,300):
            return 'PROTECTION_AMBIGUOUS'
        if elapsed_s>=JD_READY_TIMEOUT_S:
            return 'TRANSITION_TIMEOUT'
        if (jd_accepted and mode==get_mode=='JOINT_DEFAULT' and status==get_status==100
            and fsm==1 and body==1):
            return 'JD_READY'
        return 'TRANSITION_TIMEOUT' if elapsed_s>=JD_READY_TIMEOUT_S else 'TRANSITION_PENDING'
    if phase in ('SD','HOLD'):
        # SD is entered ONLY after a positively verified JD/FSM1/body1 barrier.
        if fsm==6 or mode in ('PASSIVE_DEFAULT','DAMPING_DEFAULT'):
            return 'PROTECTION_CONFIRMED'
        if (mode not in ('JOINT_DEFAULT','STAND_DEFAULT') or get_mode not in ('JOINT_DEFAULT','STAND_DEFAULT')
            or fsm not in (1,4) or body!=1 or status not in (100,200,300) or get_status not in (100,200,300)):
            return 'PROTECTION_AMBIGUOUS'
        if phase=='SD' and elapsed_s>=SD_READY_TIMEOUT_S:
            return 'TRANSITION_TIMEOUT'
        if sd_accepted and active_mode_agrees(common,readback):
            return 'SD_READY'
        return 'TRANSITION_PENDING'
    return 'TRANSITION_PENDING'


class StandObserver(ResetOnlyObserver):
    def __init__(self, output_dir, reset_event_path, protocol='sd'):
        if protocol not in ('sd','jd-sd'):
            raise ValueError('unsupported standing protocol')
        self.protocol=protocol
        self.unique_topic_stamps={}
        self.duplicate_topic_counts={}
        super().__init__(output_dir,reset_event_path)
        self.sd_client = self.create_client(SetMcAction, SET_SERVICE)
        self.get_client = self.create_client(GetMcAction, GET_SERVICE)
        self.sd_count = 0
        self.sd_future = None
        self.sd_response = None
        self.get_future = None
        self.get_result = None
        self.get_received_ns = None
        self.get_pending_request_ns = None
        self.get_result_request_ns = None
        self.next_get_ns = 0
        self.hold_start_ns = None
        self.last_hold_ns = None
        self.hold_samples = 0
        self.hold_resets = 0
        self.first_active_ns = None
        self.sd_request_ns = None
        self.last_unique_ns = None
        self.jd_count=0
        self.jd_request_ns=None
        self.jd_future=None
        self.jd_response=None
        self.jd_ready_ns=None
        self.initial_fsm6=None
        self.left_initial_fsm6=False
        self.transition_state='TRANSITION_PENDING'
        self.last_transition_state=None
        self.max_hold_s=0.
        self.velocity_graph=None
        self.velocity_graph_error=None
        self.joint_limits = json.loads((output_dir/'standing_joint_limits.json').read_text())['limits']

    def check_velocity_graph(self, *, reason):
        """Record a graph result or fail closed; absence is never inferred."""
        try:
            snapshot = velocity_graph_snapshot(self)
        except Exception as exc:
            self.velocity_graph_error = f'{type(exc).__name__}: {exc}'
            self._write({'event': 'VELOCITY_GRAPH_ERROR', 'reason': reason,
                         'receipt_monotonic_ns': time.monotonic_ns(),
                         'error': self.velocity_graph_error})
            self.failure = 'VELOCITY_GRAPH_INSPECTION_ERROR:' + reason
            return None
        self.velocity_graph = snapshot
        self._write({'event': 'VELOCITY_GRAPH', 'reason': reason,
                     'receipt_monotonic_ns': time.monotonic_ns(),
                     'snapshot': snapshot})
        if snapshot['publisher_count'] != 0:
            self.failure = 'VELOCITY_PUBLISHER_CONFLICT:' + reason
            return None
        return snapshot

    def _callback(self,key):
        receive=super()._callback(key)
        def advancing_receive(message):
            stamp=_stamp_ns(message)
            if self.protocol=='jd-sd':
                previous=self.unique_topic_stamps.get(key)
                progression=source_progress(previous,stamp)
                if progression=='MISSING':
                    self.failure='MISSING_SOURCE_STAMP:'+key
                    return
                if progression in ('DUPLICATE','REGRESSION'):
                    self._write({'event':'NONADVANCING_TOPIC_SAMPLE','topic_key':key,
                                 'source_stamp_ns':stamp,'previous_stamp_ns':previous,
                                 'receipt_monotonic_ns':time.monotonic_ns(),'raw':_safe(message)})
                    if progression=='REGRESSION':self.failure='SOURCE_STAMP_REGRESSION:'+key
                    else:self.duplicate_topic_counts[key]=self.duplicate_topic_counts.get(key,0)+1
                    return
                self.unique_topic_stamps[key]=stamp
            receive(message)
        return advancing_receive

    def joint_fault(self):
        state = self.latest_messages.get('leg_state')
        joints = list(state.joints) if state is not None else []
        by_name = {j.name:j for j in joints}
        if len(joints)!=12 or set(by_name)!=set(self.joint_limits):
            return 'LEG_SOURCE_LIMIT_MAPPING_INCOMPLETE'
        for name,joint in by_name.items():
            limit=self.joint_limits[name]
            if not math.isfinite(joint.position) or not math.isfinite(joint.velocity):
                return 'NONFINITE_JOINT:'+name
            if not limit['lower_position']<=joint.position<=limit['upper_position']:
                return 'SOURCE_JOINT_POSITION_LIMIT:'+name
            if abs(joint.velocity)>limit['max_velocity']:
                return 'SOURCE_JOINT_VELOCITY_LIMIT:'+name
            if joint.error_code:
                return 'JOINT_ERROR_CODE:'+name
        return None

    def _consider_odom(self,message,receipt_ns,stamp_ns):
        self._load_reset_event()
        previous = self.last_odom_stamp_ns
        super()._consider_odom(message,receipt_ns,stamp_ns)
        if self.reset_ns is not None and previous is not None and stamp_ns is not None and stamp_ns<previous:
            self.failure='POST_RESET_ODOM_STAMP_REGRESSION'
        if self.reset_ns is None or self.failure or self.complete:
            return
        advancing = stamp_ns is not None and (previous is None or stamp_ns > previous)
        if not advancing:
            return
        self.last_unique_ns = receipt_ns
        row = self.post_reset_rows[-1] if self.post_reset_rows else None
        if row is None or row['receipt_monotonic_ns'] != receipt_ns:
            return
        pose = row['pose']
        if self.upright_streak>=3 or self.sd_count or self.jd_count:
            fault=self.joint_fault()
            if fault:
                self.failure=fault
                return
        effect = classify_reset_effect(self.pre_reset_snapshot,self.post_reset_rows)
        common = self.latest_messages.get('common')
        raw = _safe(common) if common is not None else {}
        if self.protocol=='jd-sd' and (self.jd_count or self.sd_count) and raw.get('input_source',{}).get('name'):
            self.failure='CONFLICTING_ACTIVE_INPUT_SOURCE';return
        command_raw=_safe(self.latest_messages.get('leg_command')) or {}
        if self.protocol=='jd-sd' and self.upright_streak>=3:
            cj=command_raw.get('joints',[])
            if len(cj)!=12 or {j.get('name') for j in cj}!=set(self.joint_limits):
                self.failure='INVALID_LEG_COMMAND_MAPPING';return
            for j in cj:
                if (any(not isinstance(j.get(k),(int,float)) or not math.isfinite(j[k])
                        for k in ('position','velocity','effort','stiffness','damping'))
                    or j['stiffness']<0 or j['damping']<0):
                    self.failure='INVALID_STANDING_COMMAND:'+j['name'];return
        gains_active=any(abs(j.get(k,0))>1e-8 for j in command_raw.get('joints',[])
                         for k in ('stiffness','damping','effort'))
        qualified_jd=False
        if self.protocol=='jd-sd' and self.jd_count:
            phase=('HOLD' if self.first_active_ns is not None else 'SD') if self.sd_count else 'JD'
            elapsed=(receipt_ns-(self.sd_request_ns if self.sd_count else self.jd_request_ns))/1e9
            state=native_transition_state(phase,raw,self.get_result or {},elapsed_s=elapsed,
                jd_accepted=bool(self.jd_response and self.jd_response['accepted']),
                sd_accepted=bool(self.sd_response and self.sd_response['accepted']),
                initial_fsm6=self.initial_fsm6,left_initial_fsm6=self.left_initial_fsm6)
            if raw.get('fsm_state',{}).get('current_state') in (1,4):
                self.left_initial_fsm6=True
            self.transition_state=state
            if state!=self.last_transition_state:
                self._write({'event':'NATIVE_TRANSITION','receipt_monotonic_ns':receipt_ns,
                             'phase':phase,'state':state,'elapsed_s':elapsed,
                             'raw_common':raw,'get_readback':self.get_result})
                self.last_transition_state=state
            if state in ('PROTECTION_CONFIRMED','PROTECTION_AMBIGUOUS','TRANSITION_TIMEOUT'):
                self.failure=phase+':'+state
                return
            if state=='JD_READY' and gains_active and self._fresh(receipt_ns) and safe_posture(pose):
                if (self.get_received_ns is not None and self.get_result_request_ns is not None
                    and self.reset_ns is not None and self.get_received_ns >= self.reset_ns
                    and self.get_result_request_ns >= self.reset_ns
                    and self.received_ns.get('common', 0) >= self.reset_ns
                    and 0<=(receipt_ns-self.get_received_ns)/1e9<=MAX_STATE_AGE_S):
                    self.jd_ready_ns=self.jd_ready_ns or receipt_ns
                    qualified_jd=True
        action=None
        if self.protocol=='jd-sd':
            if not self.jd_count: action='JOINT_DEFAULT'
            elif not self.sd_count and qualified_jd:
                action='STAND_DEFAULT'
        elif not self.sd_count:
            action='STAND_DEFAULT'
        # Three advancing, fresh upright samples plus actual Reset effect; no
        # delayed host service discovery or disk-based trigger after this gate.
        if action is not None and self.upright_streak >= 3 and request_allowed(
            reset_verified=self.reset_event is not None,
            reset_effect=effect['status']=='RESET_EFFECT_OBSERVED',
            fresh=self._fresh(receipt_ns),pose=pose,advancing=advancing,
            count=self.jd_count if action=='JOINT_DEFAULT' else self.sd_count,
        ):
            if self.protocol == 'jd-sd' and self.check_velocity_graph(
                reason='ACTION_GATE_' + action
            ) is None:
                return
            if not self.sd_client.service_is_ready():
                self.failure = 'SD_SERVICE_DISAPPEARED_AT_SAFE_REQUEST_GATE'
                return
            desc = common.action_info.action_desc if common is not None else None
            expected='JOINT_DEFAULT' if self.protocol=='jd-sd' and action=='STAND_DEFAULT' else 'PASSIVE_DEFAULT'
            if desc != expected:
                self.failure = 'UNEXPECTED_PRE_SD_MODE:'+str(desc)
                return
            if (self.get_received_ns is None or self.get_result_request_ns is None
                or self.reset_ns is None or self.get_received_ns < self.reset_ns
                or self.get_result_request_ns < self.reset_ns
                or self.received_ns.get('common', 0) < self.reset_ns
                or not 0<=(receipt_ns-self.get_received_ns)/1e9<=MAX_STATE_AGE_S
                or (self.get_result or {}).get('info',{}).get('action_desc')!=expected
                or (self.get_result or {}).get('info',{}).get('status',{}).get('value')!=100):
                self.failure='UNQUALIFIED_PRE_ACTION_GET_READBACK';return
            if common.input_source.name:
                self.failure = 'CONFLICTING_ACTIVE_INPUT_SOURCE'
                return
            request = SetMcAction.Request()
            request.header.stamp = self.get_clock().now().to_msg()
            request.source = 'node.set_mc_action'
            request.command.action_desc = action
            # Re-evaluate host age immediately before the sole service call.
            now = time.monotonic_ns()
            if (now-receipt_ns)/1e9 > .05 or not self._fresh(now):
                self.failure = 'TAKEOVER_TOO_LATE_FRESH_REQUEST_GATE'
                return
            if action=='JOINT_DEFAULT':
                self.jd_count=1;self.jd_request_ns=now
                self.initial_fsm6=raw.get('fsm_state',{}).get('current_state')==6
            else:
                self.sd_count = 1
                self.sd_request_ns = now
            self._write({'event':'JD_REQUEST' if action=='JOINT_DEFAULT' else 'SD_REQUEST','receipt_monotonic_ns':now,
                         'request':_safe(request),'request_pose':pose,
                         'pose_age_s':(now-receipt_ns)/1e9,
                         'state_ages_s':self._ages(now)})
            future = self.sd_client.call_async(request)
            if action=='JOINT_DEFAULT':self.jd_future=future
            else:self.sd_future=future
        if self.first_upright_ns is not None and not safe_posture(pose):
            self.failure = 'POST_RESET_UNSAFE_' + str(posture_safety_reason(pose))
            return
        common = self.latest_messages.get('common')
        commands = _safe(self.latest_messages.get('leg_command'))
        command_joints = commands.get('joints',[]) if isinstance(commands,dict) else []
        if len(command_joints)!=12 or {j['name'] for j in command_joints}!=set(self.joint_limits):
            self.failure='INVALID_LEG_COMMAND_MAPPING'
            return
        for joint in command_joints:
            if (any(not isinstance(joint.get(k),(int,float)) or not math.isfinite(joint[k])
                    for k in ('position','velocity','effort','stiffness','damping'))
                or joint['stiffness']<0 or joint['damping']<0):
                self.failure='INVALID_STANDING_COMMAND:'+joint['name']
                return
        def has_active(value):
            if isinstance(value,dict):
                return any((k in ('stiffness','damping','effort') and isinstance(v,(int,float))
                            and math.isfinite(v) and abs(v)>1e-8) or has_active(v)
                           for k,v in value.items())
            if isinstance(value,list):
                return any(has_active(v) for v in value)
            return False
        raw = _safe(common) if common is not None else {}
        if self.sd_count and raw.get('action_info',{}).get('action_desc') not in (
                'PASSIVE_DEFAULT','JOINT_DEFAULT','STAND_DEFAULT'):
            self.failure='UNRELATED_MODE_DURING_SD_TAKEOVER'
            return
        if (self.protocol=='sd' and self.sd_count and raw.get('action_info',{}).get('action_desc')=='STAND_DEFAULT'
            and raw.get('fsm_state',{}).get('current_state')==6):
            # The category is unsafe for acceptance, but its unchanged raw 6
            # during a target switch alone does not prove a NEW protection trip.
            self.failure='SD_READBACK_CONFLICT_RAW_FSM6'
            return
        active = (
            self.sd_response is not None and self.sd_response['accepted']
            and self.get_result is not None
            and active_mode_agrees(raw,self.get_result)
            and self.get_received_ns is not None
            and 0 <= (receipt_ns-self.get_received_ns)/1e9 <= MAX_STATE_AGE_S
            and self._fresh(receipt_ns) and safe_posture(pose) and has_active(commands)
        )
        if active:
            if self.hold_start_ns is None:
                self.hold_start_ns = receipt_ns
                self.hold_samples = 0
            self.hold_samples += 1
            self.last_hold_ns = receipt_ns
            self.max_hold_s=max(self.max_hold_s,(receipt_ns-self.hold_start_ns)/1e9)
            self.first_active_ns = self.first_active_ns or receipt_ns
            if (receipt_ns-self.hold_start_ns)/1e9 >= STABLE_HOLD_S and self.hold_samples>=25:
                self._finish('CAPTURE_COMPLETE')
        elif self.hold_start_ns is not None:
            self.hold_resets += 1
            self.hold_start_ns = None
            self.hold_samples = 0

    def poll_services(self,now):
        if self.get_future is not None and self.get_future.done():
            response = self.get_future.result()
            self.get_result = _safe(response)
            self.get_received_ns = now
            self.get_result_request_ns = self.get_pending_request_ns
            self._write({'event':'GET_MC_ACTION','receipt_monotonic_ns':now,'response':self.get_result})
            self.get_future = None
            self.get_pending_request_ns = None
            if response.header.code!=0:
                self.failure='GET_MC_ACTION_REJECTED'
        if (self.get_future is not None and self.get_pending_request_ns is not None
                and now - self.get_pending_request_ns > int(MAX_STATE_AGE_S * 1e9)):
            self.failure='GET_MC_ACTION_TIMEOUT'
            return
        if self.get_future is None and now>=self.next_get_ns and self.get_client.service_is_ready():
            request = GetMcAction.Request()
            request.request.header.stamp = self.get_clock().now().to_msg()
            self.get_future = self.get_client.call_async(request)
            self.get_pending_request_ns = now
            self.next_get_ns = now+50_000_000
        if self.jd_future is not None and self.jd_future.done() and self.jd_response is None:
            response=self.jd_future.result()
            accepted=response.response.header.code==0 and response.response.status.value==CommonState.SUCCESS
            self.jd_response={'raw':_safe(response),'accepted':accepted,'receipt_monotonic_ns':now}
            self._write({'event':'JD_RESPONSE',**self.jd_response})
            if not accepted:self.failure='JD_REJECTED_WITH_EXACT_RAW_RESPONSE'
        if self.sd_future is not None and self.sd_future.done() and self.sd_response is None:
            response = self.sd_future.result()
            raw = _safe(response)
            accepted = response.response.header.code==0 and response.response.status.value==CommonState.SUCCESS
            self.sd_response = {'raw':raw,'accepted':accepted,'receipt_monotonic_ns':now}
            self._write({'event':'SD_RESPONSE',**self.sd_response})
            if not accepted:
                self.failure = 'SD_REJECTED_WITH_EXACT_RAW_RESPONSE'

    def _finish(self,status,reason=None):
        if self.complete:
            return
        super()._finish(status,reason)
        _write_new_json(self.output_dir/'stand_takeover_summary.json',{
            'status':'PASS' if status=='CAPTURE_COMPLETE' and self.failure is None else 'BLOCKED',
            'failure':self.failure,'sd_requests':self.sd_count,'sd_request_monotonic_ns':self.sd_request_ns,
            'sd_response':self.sd_response,'last_get_mc_action':self.get_result,
            'protocol':self.protocol,'jd_requests':self.jd_count,'jd_response':self.jd_response,
            'jd_request_monotonic_ns':self.jd_request_ns,'jd_ready_monotonic_ns':self.jd_ready_ns,
            'transition_state':self.transition_state,'maximum_stable_hold_s':self.max_hold_s,
            'duplicate_topic_counts':self.duplicate_topic_counts,
            'velocity_graph':self.velocity_graph,
            'velocity_graph_error':self.velocity_graph_error,
            'first_active_monotonic_ns':self.first_active_ns,
            'stable_hold_s':(self.last_hold_ns-self.hold_start_ns)/1e9
                if self.last_hold_ns is not None and self.hold_start_ns is not None else 0.,
            'hold_samples':self.hold_samples,'hold_resets':self.hold_resets,
            'velocity_publishers':len(list(self.publishers)),
            'service_clients':len(list(self.clients)),
            'fsm_contract':'same pinned MC build: STABLE=4, body=1, action-ready=100; raw retained',
            'complete_raw_telemetry':'reset_only_state_trace.jsonl',
            'snapshot_at_stop':self._snapshot(),
            'protection_reason':'NOT_EXPOSED; raw 6 remains fail-closed, no inferred cause',
        })

    def run_capture(self):
        deadline = self.started_ns+15_000_000_000
        while rclpy.ok() and not self.complete:
            rclpy.spin_once(self,timeout_sec=.002)
            now=time.monotonic_ns()
            self.poll_services(now)
            if self.ready_ns is None:
                if (len(self.latest)==len(TOPIC_TYPES) and self._fresh(now)
                    and self.sd_client.service_is_ready() and self.get_client.service_is_ready()
                    and self.get_result is not None):
                    counts=self.interface_counts()
                    if counts['publishers']!=0 or counts['service_clients']!=2:
                        self._finish('BLOCKED','UNEXPECTED_OBSERVER_INTERFACE_SURFACE');break
                    if self.protocol == 'jd-sd' and self.check_velocity_graph(
                        reason='READY'
                    ) is None:
                        self._finish('BLOCKED',self.failure);break
                    if self.protocol=='jd-sd' and self.velocity_graph['status']!='PASS_ZERO_VELOCITY_PUBLISHERS':
                        self._finish('BLOCKED','VELOCITY_GRAPH_NOT_QUALIFIED');break
                    self.ready_ns=now
                    self.pre_reset_snapshot=self._snapshot(now)
                    _write_new_json(self.output_dir/'reset_watch_ready.json',{
                        'status':'READY','publishers_created':0,'service_clients_created':2,
                        'velocity_graph':self.velocity_graph,
                        'state_age_s':self._ages(now),'snapshot':self.pre_reset_snapshot})
                    _write_new_json(self.output_dir/'reset_only_pre_reset_snapshot.json',{
                        'snapshot_monotonic_ns':now,'snapshot':self.pre_reset_snapshot})
                elif now-self.started_ns>=5_000_000_000:
                    self._finish('BLOCKED','SD_CLIENT_OR_TELEMETRY_NOT_READY_BEFORE_RESET');break
            if self.ready_ns is not None:
                if not self._fresh(now):
                    self._finish('BLOCKED','CRITICAL_TELEMETRY_STALE');break
                if self.reset_ns is None:
                    _write_snapshot_replace(self.output_dir/'reset_only_latest_snapshot.json',{
                        'snapshot':self._snapshot(now)})
                self._load_reset_event()
                if (self.first_upright_ns is not None and self.last_unique_ns is not None
                    and now-self.last_unique_ns>int(MAX_STATE_AGE_S*1e9)):
                    self.failure='UNIQUE_ODOM_STALE'
            if self.failure:
                self._finish('BLOCKED',self.failure);break
            post_limit_s=7.0 if self.protocol=='jd-sd' else 5.0
            if self.reset_ns is not None and now-self.reset_ns>int(post_limit_s*1e9):
                self._finish('BLOCKED','ACTIVE_STANDING_NOT_QUALIFIED_WITHIN_BOUNDED_WINDOW');break
            if now>=deadline:
                self._finish('BLOCKED','BOUNDED_OBSERVER_DEADLINE');break
        return 0 if self.failure is None and self.complete else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reset-event-file',type=Path,required=True)
    p.add_argument('--construction-smoke',action='store_true')
    p.add_argument('--protocol',choices=('sd','jd-sd'),default='sd')
    args=p.parse_args()
    rclpy.init()
    node=None
    try:
        node=StandObserver(args.output_dir,args.reset_event_file,protocol=args.protocol)
        if args.construction_smoke:
            counts=node.interface_counts()
            if args.protocol=='jd-sd':counts['velocity_graph']=velocity_graph_snapshot(node)
            counts['status']='PASS_CONSTRUCTION_SMOKE' if counts['publishers']==0 and counts['service_clients']==2 and counts['subscriptions']==5 else 'BLOCKED'
            _write_new_json(args.output_dir/'standing_observer_smoke.json',counts)
            return 0 if counts['publishers']==0 and counts['service_clients']==2 and counts['subscriptions']==5 else 2
        return node.run_capture()
    except Exception as exc:
        _write_new_json(args.output_dir/'reset_only_observer_startup_error.json',{
            'status':'BLOCKED','error':repr(exc)})
        return 2
    finally:
        if node is not None:
            node.trace.close();node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    raise SystemExit(main())
