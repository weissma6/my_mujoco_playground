"""Reusable Robotiq Hand-E wrapper for the UR3 pick task (PolyScope X).

On PolyScope X the Hand-E is driven through the Robotiq URCapX XML-RPC server at
``http://<host>:49999/`` (NOT the legacy 63352 socket -- that socket client,
``robots/random_sample_code/robotiq_gripper.py``, is the PolyScope 5 / CB-series
path and is not used here). Native units are PERCENT (0-100). This Hand-E is
slaveId 9 ("Gripper ID 1" in the UI).

It centralises the sim<->native mapping so the same object is usable from the
Hand-E mapping notebook and from ur3_realrobot_pickloop.py.

Direction conventions (verified, see the module CLAUDE.md / the mapping notebook):
  - SIM finger position is in meters per finger, [0, 0.025]: 0 = OPEN, 0.025 = CLOSED
    (UR3Pick: finger_open = tanh(finger_touch_dist/0.05); fingers close as qpos rises).
  - NATIVE Robotiq URCapX percent, [0, 100]: default 0 % = OPEN, 100 % = CLOSED.
    Same direction as sim -> the mapping is pure scale, NO sign flip. The percent
    direction is the one bit not provable from code; confirm/flip NATIVE_OPEN_PCT /
    NATIVE_CLOSED_PCT with the open/close test in the mapping notebook.

(NOTE: the inline send_gripper() in ur3_realrobot_dependencies.py uses an INVERTED
norm->percent mapping with a "0 = closed" docstring; that is a latent bug in that
file. This wrapper gets the direction right and the pick loop routes the gripper
through here instead.)
"""

import xmlrpc.client


class HandEGripper:
    """Robotiq Hand-E via the PolyScope X URCapX XML-RPC server (percent units)."""

    # --- sim range (meters, per finger) ---
    SIM_OPEN_M = 0.0
    SIM_CLOSED_M = 0.025

    # --- native range (Robotiq URCapX percent) ---
    # Flip these two if the open/close test shows the gripper reports ~100 % when
    # open and ~0 % when closed. Nothing else in this class encodes direction.
    NATIVE_OPEN_PCT = 0.0
    NATIVE_CLOSED_PCT = 100.0

    def __init__(
        self,
        host: str,
        port: int = 49999,
        slave_id: int = 9,
        speed_pct: int = 100,
        force_pct: int = 50,
    ):
        self.host = host
        self.port = int(port)
        self.slave_id = int(slave_id)
        self.speed_pct = int(speed_pct)
        self.force_pct = int(force_pct)
        self._g = None  # xmlrpc.client.ServerProxy once connected

    # -------------------------------------------------------------------------
    # Connection / activation
    # -------------------------------------------------------------------------

    def connect(self, reset: bool = False, speed: int = None, force: int = None):
        """Connect to the URCapX XML-RPC server and activate the gripper.

        reset=True forces a full (re)activation; otherwise activateIfRequired().
        Errors propagate so a missing server / URCapX fails loudly. Activation does
        NOT auto-calibrate (no surprise full open/close sweep on connect).
        """
        g = xmlrpc.client.ServerProxy(f"http://{self.host}:{self.port}/")
        sid = self.slave_id
        if reset:
            g.activate([sid], True)
        else:
            g.activateIfRequired([sid])
        if speed is not None:
            self.speed_pct = int(speed)
        if force is not None:
            self.force_pct = int(force)
        g.setSpeed([sid], self.speed_pct)
        g.setForce([sid], self.force_pct)
        self._g = g
        print(
            f"[HandE] XML-RPC {self.host}:{self.port} slaveId={sid} "
            f"connected={g.isGripperConnected(sid)} "
            f"activated={g.isGripperActivated(sid)}"
        )
        return g

    def disconnect(self):
        """Drop the XML-RPC handle (ServerProxy holds no persistent socket).

        Leaves the gripper activated on the controller, matching the arm-side
        UR3RealRobotPick.disconnect() convention.
        """
        self._g = None

    def is_connected(self) -> bool:
        if self._g is None:
            return False
        try:
            return bool(self._g.isGripperConnected(self.slave_id))
        except Exception:
            return False

    def _require(self):
        if self._g is None:
            raise RuntimeError(
                "Gripper not connected. Call connect() before commanding/reading."
            )
        return self._g

    # -------------------------------------------------------------------------
    # Mapping (the single source of truth for sim <-> native direction)
    # -------------------------------------------------------------------------

    @classmethod
    def sim_to_native(cls, sim_m: float) -> float:
        """Map a sim finger position [0, 0.025] m to native percent [0, 100], clamped."""
        sim = min(max(float(sim_m), cls.SIM_OPEN_M), cls.SIM_CLOSED_M)
        frac = (sim - cls.SIM_OPEN_M) / (cls.SIM_CLOSED_M - cls.SIM_OPEN_M)
        pct = cls.NATIVE_OPEN_PCT + frac * (cls.NATIVE_CLOSED_PCT - cls.NATIVE_OPEN_PCT)
        return float(min(max(pct, 0.0), 100.0))

    @classmethod
    def native_to_sim(cls, pct: float) -> float:
        """Map a native percent [0, 100] back to a sim finger position [0, 0.025] m, clamped."""
        p = min(max(float(pct), 0.0), 100.0)
        span = cls.NATIVE_CLOSED_PCT - cls.NATIVE_OPEN_PCT
        frac = (p - cls.NATIVE_OPEN_PCT) / span if span != 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        sim = cls.SIM_OPEN_M + frac * (cls.SIM_CLOSED_M - cls.SIM_OPEN_M)
        return float(sim)

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    def command(self, sim_value: float, speed: int = None, force: int = None):
        """Move the gripper to a sim finger position [0, 0.025] m (0 open, 0.025 closed).

        speed/force override the per-call values; defaults persist from connect().
        """
        g = self._require()
        if speed is not None:
            self.speed_pct = int(speed)
            g.setSpeed([self.slave_id], self.speed_pct)
        if force is not None:
            self.force_pct = int(force)
            g.setForce([self.slave_id], self.force_pct)
        pct = self.sim_to_native(sim_value)
        g.move([self.slave_id], pct, 0, [0] * 16)
        return pct

    def open_gripper(self):
        """Fully open (sim 0). Direction-independent URCapX call -- correct even
        before the open/closed-pct constants are confirmed."""
        self._require().openGripper(self.slave_id)

    def close_gripper(self):
        """Fully close (sim 0.025). Direction-independent URCapX call (see open_gripper)."""
        self._require().closeGripper(self.slave_id)

    # -------------------------------------------------------------------------
    # State readback
    # -------------------------------------------------------------------------

    def read_state(self) -> dict:
        """Read the Hand-E state via XML-RPC (no motion command sent).

        Returns native values plus a sim-mapped finger position:
          pos_pct     getCurrentPosition (0-100)
          obj_flag    getObjectDetectionFlag (0 none, 1 on-open, 2 on-close)
          grasped     obj_flag in (1, 2)  -- an object stopped the fingers
          sim_finger  native_to_sim(pos_pct)  (per-finger meters, [0, 0.025])
          fault       getFault (0 = ok)
          activated   isGripperActivated (bool)
          connected   isGripperConnected (bool)
        """
        g = self._require()
        sid = self.slave_id
        pos_pct = float(g.getCurrentPosition(sid, 0, 0, 0, 0, 0))
        obj_flag = int(g.getObjectDetectionFlag(sid))
        return {
            "pos_pct": pos_pct,
            "obj_flag": obj_flag,
            "grasped": obj_flag in (1, 2),
            "sim_finger": self.native_to_sim(pos_pct),
            "fault": int(g.getFault(sid)),
            "activated": bool(g.isGripperActivated(sid)),
            "connected": bool(g.isGripperConnected(sid)),
        }
