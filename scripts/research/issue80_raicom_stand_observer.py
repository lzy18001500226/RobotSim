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
    classify_reset_effect,
)

SET_SERVICE = '/aimdk_5Fmsgs/srv/SetMcAction'
GET_SERVICE = '/aimdk_5Fmsgs/srv/GetMcAction'
STABLE_HOLD_S = 2.5
MAX_PLANAR_SPEED_M_S = .20


def safe_posture(pose):
    try:
        values = [float(pose[k]) for k in ('z_m','roll_rad','pitch_rad')]
        frame_values = [float(pose[k]) for k in ('x_m','y_m','yaw_rad')]
        velocity = [float(v) for v in pose['linear_velocity_m_s']]
        return (all(math.isfinite(v) for v in values+velocity+frame_values)
                and values[0] >= MIN_ROOT_Z_M
                and max(abs(values[1]),abs(values[2])) < MAX_ABS_TILT_RAD
                and math.hypot(*velocity[:2]) <= MAX_PLANAR_SPEED_M_S)
    except (KeyError,TypeError,ValueError):
        return False


def request_allowed(*, reset_verified, reset_effect, fresh, pose, advancing, count):
    return (reset_verified and reset_effect and fresh and safe_posture(pose)
            and advancing and count == 0)


def active_mode_agrees(common, readback):
    return (readback.get('header',{}).get('code')==0
            and readback.get('info',{}).get('action_desc')=='STAND_DEFAULT'
            and readback.get('info',{}).get('status',{}).get('value')==100
            and common.get('action_info',{}).get('action_desc')=='STAND_DEFAULT'
            and common.get('action_info',{}).get('status',{}).get('value')==100
            and common.get('body_status',{}).get('value')==1
            and common.get('fsm_state',{}).get('current_state')==4)


class StandObserver(ResetOnlyObserver):
    def __init__(self, output_dir, reset_event_path):
        super().__init__(output_dir,reset_event_path)
        self.sd_client = self.create_client(SetMcAction, SET_SERVICE)
        self.get_client = self.create_client(GetMcAction, GET_SERVICE)
        self.sd_count = 0
        self.sd_future = None
        self.sd_response = None
        self.get_future = None
        self.get_result = None
        self.get_received_ns = None
        self.next_get_ns = 0
        self.hold_start_ns = None
        self.last_hold_ns = None
        self.hold_samples = 0
        self.hold_resets = 0
        self.first_active_ns = None
        self.sd_request_ns = None
        self.last_unique_ns = None
        self.joint_limits = json.loads((output_dir/'standing_joint_limits.json').read_text())['limits']

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
        if self.upright_streak>=3 or self.sd_count:
            fault=self.joint_fault()
            if fault:
                self.failure=fault
                return
        effect = classify_reset_effect(self.pre_reset_snapshot,self.post_reset_rows)
        # Three advancing, fresh upright samples plus actual Reset effect; no
        # delayed host service discovery or disk-based trigger after this gate.
        if self.sd_count == 0 and self.upright_streak >= 3 and request_allowed(
            reset_verified=self.reset_event is not None,
            reset_effect=effect['status']=='RESET_EFFECT_OBSERVED',
            fresh=self._fresh(receipt_ns),pose=pose,advancing=advancing,count=self.sd_count,
        ):
            if not self.sd_client.service_is_ready():
                self.failure = 'SD_SERVICE_DISAPPEARED_AT_SAFE_REQUEST_GATE'
                return
            common = self.latest_messages.get('common')
            desc = common.action_info.action_desc if common is not None else None
            if desc != 'PASSIVE_DEFAULT':
                self.failure = 'UNEXPECTED_PRE_SD_MODE:'+str(desc)
                return
            if common.input_source.name:
                self.failure = 'CONFLICTING_ACTIVE_INPUT_SOURCE'
                return
            request = SetMcAction.Request()
            request.header.stamp = self.get_clock().now().to_msg()
            request.source = 'node.set_mc_action'
            request.command.action_desc = 'STAND_DEFAULT'
            # Re-evaluate host age immediately before the sole service call.
            now = time.monotonic_ns()
            if (now-receipt_ns)/1e9 > .05 or not self._fresh(now):
                self.failure = 'TAKEOVER_TOO_LATE_FRESH_REQUEST_GATE'
                return
            self.sd_count = 1
            self.sd_request_ns = now
            self._write({'event':'SD_REQUEST','receipt_monotonic_ns':now,
                         'request':_safe(request),'request_pose':pose,
                         'pose_age_s':(now-receipt_ns)/1e9,
                         'state_ages_s':self._ages(now)})
            self.sd_future = self.sd_client.call_async(request)
        if self.first_upright_ns is not None and not safe_posture(pose):
            self.failure = 'TAKEOVER_TOO_LATE_OR_ACTIVE_POSTURE_UNSAFE'
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
        if (self.sd_count and raw.get('action_info',{}).get('action_desc')=='STAND_DEFAULT'
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
            self._write({'event':'GET_MC_ACTION','receipt_monotonic_ns':now,'response':self.get_result})
            self.get_future = None
            if response.header.code!=0:
                self.failure='GET_MC_ACTION_REJECTED'
        if self.get_future is None and now>=self.next_get_ns and self.get_client.service_is_ready():
            request = GetMcAction.Request()
            request.request.header.stamp = self.get_clock().now().to_msg()
            self.get_future = self.get_client.call_async(request)
            self.next_get_ns = now+50_000_000
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
                    self.ready_ns=now
                    self.pre_reset_snapshot=self._snapshot(now)
                    _write_new_json(self.output_dir/'reset_watch_ready.json',{
                        'status':'READY','publishers_created':0,'service_clients_created':2,
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
            if self.reset_ns is not None and now-self.reset_ns>5_000_000_000:
                self._finish('BLOCKED','ACTIVE_STANDING_NOT_QUALIFIED_WITHIN_5S');break
            if now>=deadline:
                self._finish('BLOCKED','BOUNDED_OBSERVER_DEADLINE');break
        return 0 if self.failure is None and self.complete else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reset-event-file',type=Path,required=True)
    p.add_argument('--construction-smoke',action='store_true')
    args=p.parse_args()
    rclpy.init()
    node=None
    try:
        node=StandObserver(args.output_dir,args.reset_event_file)
        if args.construction_smoke:
            counts=node.interface_counts()
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
