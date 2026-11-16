"""Bounded native-snapshot acquisition; never qualify an expired measurement."""


def read_fresh_native_snapshot(path, *, generation=None, budget_s=1.0):
    import json
    import time
    from pathlib import Path
    if not .05<=budget_s<=1.:
        raise ValueError("native acquisition recovery budget must be between50ms and1s")
    began=time.monotonic_ns()
    deadline=began+round(budget_s*1e9)
    first_age=None
    while True:
        raw=Path(path).read_bytes()
        rows=raw.splitlines()
        if not raw.endswith(bytes([10])) and len(rows)>1:
            rows=rows[:-1]
        row=json.loads(rows[-1])
        stamp=int(row["host_monotonic_ns"]);now=time.monotonic_ns()
        age=(now-stamp)*1e-9
        if row.get("engine_version")!=337:
            raise RuntimeError("NATIVE_SNAPSHOT_ABI_CHANGED")
        if generation is not None and row["generation"]!=generation:
            raise RuntimeError("NATIVE_SNAPSHOT_GENERATION_CHANGED")
        if age<0:
            raise RuntimeError("NATIVE_SNAPSHOT_FROM_FUTURE")
        if first_age is None:first_age=age
        if age<=.20 and (first_age<=.20 or now<=deadline):
            if first_age>.20:
                print(json.dumps({"native_snapshot_recovered":True,"initial_age_s":first_age,
                    "accepted_age_s":age,"wait_s":(now-began)*1e-9}),flush=True)
            return row
        if now>=deadline:
            raise RuntimeError("NATIVE_SNAPSHOT_REACQUISITION_EXPIRED:"+json.dumps(
                {"source_age_s":age,"wait_s":(now-began)*1e-9,"budget_s":budget_s}))
        time.sleep(.002)


def generated_source():
    import inspect
    return inspect.getsource(read_fresh_native_snapshot)
