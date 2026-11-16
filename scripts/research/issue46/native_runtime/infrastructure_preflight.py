"""Read-only startup diagnostics; never repairs clocks or starts robot processes.

100 ppm and 10 ms are infrastructure diagnostic bounds, not robot physical
limits. RAW was cross-checked against Windows QPC in the preceding evidence.
A healthy short observation is necessary, not a long-term stability claim.
"""
import json
import math
import os
from pathlib import Path
import time

MIN_CLOCK_WINDOW_S = 30.0
MAX_RATE_ERROR_PPM = 100.0
MAX_CLOCK_STEP_S = .010


def assess_clock(samples, *, min_window_s=MIN_CLOCK_WINDOW_S):
    failures = []
    if len(samples) < 2 or any(not math.isfinite(float(row[k])) for row in samples for k in ('raw_s','monotonic_s','realtime_s')):
        return dict(passed=False, failures=['INSUFFICIENT_OR_INVALID_CLOCK_SAMPLES'])
    elapsed = samples[-1]['raw_s']-samples[0]['raw_s']
    if elapsed < min_window_s: failures.append('CLOCK_WINDOW_INCOMPLETE')
    monotonic_elapsed = samples[-1]['monotonic_s']-samples[0]['monotonic_s']
    rate_ppm = (monotonic_elapsed/elapsed-1.)*1e6 if elapsed>0 else None
    if rate_ppm is None or abs(rate_ppm)>MAX_RATE_ERROR_PPM: failures.append('ADJUSTED_CLOCK_RATE')
    steps=[]
    for first,last in zip(samples,samples[1:]):
        if last['raw_s']<=first['raw_s'] or last['monotonic_s']<=first['monotonic_s']:
            failures.append('NONINCREASING_ELAPSED_CLOCK')
        delta=(last['realtime_s']-first['realtime_s'])-(last['monotonic_s']-first['monotonic_s'])
        if abs(delta)>MAX_CLOCK_STEP_S:
            steps.append(dict(raw_elapsed_s=last['raw_s']-samples[0]['raw_s'],step_s=delta))
    if steps: failures.append('REALTIME_DISCONTINUITY')
    return dict(passed=not failures, failures=sorted(set(failures)), elapsed_raw_s=elapsed, rate_error_ppm=rate_ppm, steps=steps, diagnostic_bounds=dict(min_window_s=min_window_s,max_rate_error_ppm=MAX_RATE_ERROR_PPM,max_step_s=MAX_CLOCK_STEP_S))


def inspect_runtime(*, duration_s=35.0):
    try:
        master,slave=os.openpty()
        os.close(master);os.close(slave)
    except OSError as exc:
        return dict(schema='robotsim.infrastructure-preflight.v1',passed=False,failures=['PTY_UNAVAILABLE'],detail=str(exc))
    raw=getattr(time,'CLOCK_MONOTONIC_RAW',None)
    if raw is None:
        return dict(schema='robotsim.infrastructure-preflight.v1',passed=False,failures=['RAW_CLOCK_UNAVAILABLE'])
    samples=[]
    start=time.clock_gettime(raw)
    while True:
        samples.append(dict(raw_s=time.clock_gettime(raw),monotonic_s=time.monotonic(),realtime_s=time.time()))
        if samples[-1]['raw_s']-start>=duration_s:break
        time.sleep(.05)
    clock=assess_clock(samples)
    return dict(schema='robotsim.infrastructure-preflight.v1',passed=clock['passed'],failures=clock['failures'],pty='PASS',clock=clock,samples=samples,robot_processes_started=0)


def require_healthy_runtime(output):
    from scripts.agent import path_governance as pg
    report=inspect_runtime()
    pg.write_new_text(Path(output)/'infrastructure_preflight.json',json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        pg.write_new_text(Path(output)/'result.json',json.dumps(dict(schema='robotsim.issue46.xvfb-native-reset-stand.v1',gates={'INFRASTRUCTURE_PREFLIGHT':'FAIL','SIMULATOR':'NOT_RUN','MC':'NOT_RUN'},failure={'gate':'INFRASTRUCTURE_PREFLIGHT','detail':report['failures']},physical_episode_started=False),indent=2)+'\n')
        raise RuntimeError('INFRASTRUCTURE_PREFLIGHT:'+','.join(report['failures']))
    return report


def main():
    from scripts.agent import path_governance as pg
    output=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='infrastructure-preflight').native_path
    try: result=require_healthy_runtime(output)
    except RuntimeError as exc:
        print(json.dumps(dict(output=str(output),passed=False,error=str(exc))));return 2
    print(json.dumps(dict(output=str(output),passed=result['passed'])));return 0


if __name__=='__main__':raise SystemExit(main())
