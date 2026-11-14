"""Bounded mode verification for research clients; no motion publishers.

A service reply and a confirmed mode are separate observations. Transport
failure never permits a second state-changing request. Physical readiness
continues to be owned by the existing posture/contact/safety gates.
"""
import time as _infra_time


def infra_elapsed():
    raw = getattr(_infra_time, "CLOCK_MONOTONIC_RAW", None)
    return _infra_time.clock_gettime(raw) if raw is not None else _infra_time.perf_counter()


class ModeHandshakeError(RuntimeError):
    pass


def call_service(node, client, request, *, timeout_s=5.0):
    """One request, one callback owner, cancellation of timed-out local futures."""
    import rclpy
    deadline = infra_elapsed() + timeout_s
    while rclpy.ok():
        remaining = deadline - infra_elapsed()
        if remaining <= 0:
            raise ModeHandshakeError("service unavailable: " + client.srv_name)
        if client.wait_for_service(timeout_sec=min(.05, remaining)):
            break
    else:
        raise ModeHandshakeError("ROS_SHUTDOWN")
    future = client.call_async(request)
    try:
        while rclpy.ok() and not future.done():
            remaining = deadline - infra_elapsed()
            if remaining <= 0:
                raise ModeHandshakeError("service timeout: " + client.srv_name)
            executor = getattr(node, "feedback_executor", None)
            if executor is None:
                owner = getattr(node, "executor", None)
                if owner is None:
                    owner = rclpy.get_global_executor()
                    owner.add_node(node)
                owner.spin_once(timeout_sec=min(.01, remaining))
            else:
                error = getattr(node, "feedback_executor_error", None)
                thread = getattr(node, "feedback_thread", None)
                if error is not None or (thread is not None and not thread.is_alive()):
                    raise ModeHandshakeError("FEEDBACK_EXECUTOR_FAILURE:" + str(error))
                _infra_time.sleep(min(.005, remaining))
        if not future.done() or future.cancelled() or future.result() is None:
            raise ModeHandshakeError("service timeout: " + client.srv_name)
        return future.result()
    finally:
        # Humble queues done-callback cleanup on the executor. Prune this
        # transaction immediately as well; the installed API is idempotent.
        client.remove_pending_request(future)
        if not future.done():
            future.cancel()


def ensure_mode(desired, read_mode, send_once, *, timeout_s=5.0):
    """Fresh RPC readback; no cached mode and no retry of SetMcAction.

    read_mode(remaining_s) returns name/status/header_code. send_once returns
    acknowledgement status/header_code. A transport timeout is ambiguous;
    only a subsequent valid applied-mode readback can resolve it.
    """
    started = infra_elapsed()
    deadline = started + timeout_s
    def remaining():
        value = deadline - infra_elapsed()
        if value <= 0:
            raise ModeHandshakeError("MODE_READBACK_TIMEOUT:" + desired)
        return value
    def read():
        row = read_mode(remaining())
        if row.get("header_code") != 0 or row.get("status") not in (100, 200):
            raise ModeHandshakeError("MODE_READBACK_INVALID:" + repr(row))
        return row
    initial = read()
    result = {"desired": desired, "initial": initial, "set_sent": False,
              "acknowledgement": None, "transport_error": None}
    # Delivered McActionStatus: IDLE=0, RUNNING=100, TRANSITION=200.
    # A matching name during TRANSITION is not an applied, running mode.
    if initial["name"] == desired and initial["status"] == 100:
        result.update(outcome="ALREADY_APPLIED", applied=initial,
                      elapsed_s=infra_elapsed()-started)
        return result
    if initial["name"] != desired:
        if initial["status"] != 100:
            raise ModeHandshakeError("MODE_TRANSITION_NOT_READY:" + repr(initial))
        result["set_sent"] = True
        try:
            ack = send_once(remaining())
        except RuntimeError as exc:
            # Only an ambiguous service timeout may be reconciled by readback.
            if "service timeout" not in str(exc):
                raise
            result["transport_error"] = str(exc)
        else:
            result["acknowledgement"] = ack
            if ack.get("header_code") != 0 or ack.get("status") != 1:
                raise ModeHandshakeError("MODE_REQUEST_REJECTED:" + repr(ack))
    while True:
        observed = read()
        if observed["name"] == desired and observed["status"] == 100:
            outcome = ("APPLIED_NO_SET" if not result["set_sent"] else
                       "APPLIED_ACK_MISSING" if result["transport_error"] else "APPLIED")
            result.update(outcome=outcome, applied=observed,
                          elapsed_s=infra_elapsed()-started)
            return result
        _infra_time.sleep(min(.05, remaining()))

# END EMBEDDED

def generated_source():
    from pathlib import Path
    return Path(__file__).read_text().split("# END EMBEDDED" + "\n")[0]
