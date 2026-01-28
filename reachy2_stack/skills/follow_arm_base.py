# # # # # # # import time
# # # # # # # import numpy as np

# # # # # # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # # # # # from reachy2_stack.core.client import ReachyClient
# # # # # # # from reachy2_stack.control.base import BaseController

# # # # # # # # ---------------- CONFIG ----------------
# # # # # # # HOST = "192.168.1.71"
# # # # # # # SIDE = "left"

# # # # # # # # Target pose (base <- ee)
# # # # # # # theta = np.deg2rad(0.0)
# # # # # # # Rz = np.array([
# # # # # # #     [np.cos(theta), -np.sin(theta), 0.0, 0.0],
# # # # # # #     [np.sin(theta),  np.cos(theta), 0.0, 0.0],
# # # # # # #     [0.0,            0.0,           1.0, 0.0],
# # # # # # #     [0.0,            0.0,           0.0, 1.0],
# # # # # # # ])
# # # # # # # A = np.array(
# # # # # # #     [
# # # # # # #         [0, 0, -1, 0.1],
# # # # # # #         [0, 1,  0, 0.4],
# # # # # # #         [1, 0,  0, 0.1],
# # # # # # #         [0, 0,  0, 1.0],
# # # # # # #     ],
# # # # # # #     dtype=float,
# # # # # # # )
# # # # # # # A = Rz @ A


# # # # # # # # Search parameters (planar)
# # # # # # # MAX_BASE_TOTAL = 3     # max distance base is allowed to travel in total (m)
# # # # # # # MAX_STEP = 0.30           # single base command clamp (m)  (your base.goto_odom limit comfort)
# # # # # # # R_STEP = 0.05             # radial step for search (m)
# # # # # # # R_MAX = 2.0               # max search radius (m) (must be <= MAX_BASE_TOTAL)
# # # # # # # ANGLES = 16               # samples per ring (increase to 24/32 if needed)

# # # # # # # ARM_DURATION = 2.0
# # # # # # # BASE_TIMEOUT = 3.0
# # # # # # # # ---------------------------------------

# # # # # # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # # # # # #     """Return True if robot accepted the command (reachable)."""
# # # # # # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # # # # # #     gid = getattr(resp, "id", None)
# # # # # # #     print("[ARM] goto id:", gid)
# # # # # # #     return gid is not None and gid != -1

# # # # # # # def apply_base_and_update(A_cur: np.ndarray, dx: float, dy: float) -> np.ndarray:
# # # # # # #     """Base moves +dx,+dy in odom => target in new base frame shifts by -dx,-dy."""
# # # # # # #     A2 = A_cur.copy()
# # # # # # #     A2[0, 3] -= dx
# # # # # # #     A2[1, 3] -= dy
# # # # # # #     return A2

# # # # # # # # --- connect ---
# # # # # # # cfg = ReachyConfig(host=HOST)
# # # # # # # client = ReachyClient(cfg)
# # # # # # # client.connect()
# # # # # # # client.turn_on_all()

# # # # # # # arm = client._get_arm(SIDE)
# # # # # # # base = BaseController(client=client, world=None)


# # # # # # # print("[BASE] Resetting odometry...")
# # # # # # # base.reset_odometry()
# # # # # # # time.sleep(0.5)

# # # # # # # A_cur = A.copy()

# # # # # # # print("\n[1] First try without moving base")
# # # # # # # if try_arm(arm, A_cur):
# # # # # # #     print("[SUCCESS] Reached without base move.")
# # # # # # #     arm.turn_off_smoothly()
# # # # # # #     print("Done.")
# # # # # # #     raise SystemExit(0)

# # # # # # # print("\n[2] Unreachable -> planar search for a base (dx,dy) that makes it reachable")
# # # # # # # total_moved = 0.0

# # # # # # # # Spiral / ring search: r = R_STEP .. R_MAX
# # # # # # # radii = np.arange(R_STEP, R_MAX + 1e-9, R_STEP)

# # # # # # # found = False
# # # # # # # best_dxdy = None

# # # # # # # for r in radii:
# # # # # # #     # sample angles uniformly
# # # # # # #     for i in range(ANGLES):
# # # # # # #         theta = 2.0 * np.pi * (i / ANGLES)
# # # # # # #         dx = float(r * np.cos(theta))
# # # # # # #         dy = float(r * np.sin(theta))

# # # # # # #         # Respect global travel budget
# # # # # # #         if np.hypot(dx, dy) > MAX_BASE_TOTAL:
# # # # # # #             continue

# # # # # # #         # Candidate pose if base moved (dx,dy) from current base frame
# # # # # # #         A_try = apply_base_and_update(A_cur, dx, dy)

# # # # # # #         print(f"\n[SEARCH] r={r:.2f} theta={theta:.2f} -> candidate base move dx={dx:.3f}, dy={dy:.3f}")
# # # # # # #         # Try arm WITHOUT moving base yet (cheap test by asking robot)
# # # # # # #         if try_arm(arm, A_try):
# # # # # # #             print("[SEARCH] Found a feasible base offset.")
# # # # # # #             found = True
# # # # # # #             best_dxdy = (dx, dy)
# # # # # # #             break

# # # # # # #     if found:
# # # # # # #         break

# # # # # # # if not found:
# # # # # # #     print("[FAIL] No feasible base offset found within search radius.")
# # # # # # #     arm.turn_off_smoothly()
# # # # # # #     print("Done.")
# # # # # # #     raise SystemExit(1)

# # # # # # # dx, dy = best_dxdy

# # # # # # # # Execute base move in possibly multiple clamped steps
# # # # # # # remaining_dx = dx
# # # # # # # remaining_dy = dy

# # # # # # # print(f"\n[BASE] Executing base move to dx={dx:.3f}, dy={dy:.3f} (may be split into steps)")
# # # # # # # while True:
# # # # # # #     step = float(np.hypot(remaining_dx, remaining_dy))
# # # # # # #     if step < 1e-6:
# # # # # # #         break

# # # # # # #     s = 1.0
# # # # # # #     if step > MAX_STEP:
# # # # # # #         s = MAX_STEP / step

# # # # # # #     sx = remaining_dx * s
# # # # # # #     sy = remaining_dy * s

# # # # # # #     print(f"[BASE] step goto_odom(x={sx:.3f}, y={sy:.3f})")
# # # # # # #     base.reset_odometry()  # you want absolute moves
# # # # # # #     base.goto_odom(
# # # # # # #         x=float(sx),
# # # # # # #         y=float(sy),
# # # # # # #         theta=0.0,
# # # # # # #         wait=True,
# # # # # # #         distance_tolerance=0.05,
# # # # # # #         angle_tolerance=5.0,
# # # # # # #         timeout=BASE_TIMEOUT,
# # # # # # #     )
# # # # # # #     total_moved += float(np.hypot(sx, sy))
# # # # # # #     if total_moved > MAX_BASE_TOTAL:
# # # # # # #         print("[FAIL] Exceeded MAX_BASE_TOTAL while moving base.")
# # # # # # #         arm.turn_off_smoothly()
# # # # # # #         raise SystemExit(1)

# # # # # # #     # update target in new base frame
# # # # # # #     A_cur = apply_base_and_update(A_cur, sx, sy)

# # # # # # #     remaining_dx -= sx
# # # # # # #     remaining_dy -= sy

# # # # # # # print("\n[3] Retrying arm.goto after base move")
# # # # # # # if try_arm(arm, A_cur):
# # # # # # #     print("[SUCCESS] Reached after base assist.")
# # # # # # # else:
# # # # # # #     print("[FAIL] Unexpected: was feasible in search, but failed after moving base (odom drift/slip?).")

# # # # # # # arm.turn_off_smoothly()
# # # # # # # reachy = client.connect_reachy
# # # # # # # reachy.mobile_base.turn_off()
# # # # # # # print("Done.")


# # # # # # #!/usr/bin/env python3
# # # # # # import time
# # # # # # import numpy as np

# # # # # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # # # # from reachy2_stack.core.client import ReachyClient
# # # # # # from reachy2_stack.control.base import BaseController

# # # # # # # ---------------- CONFIG ----------------
# # # # # # HOST = "192.168.1.71"
# # # # # # SIDE = "right"

# # # # # # # --- Target pose (base <- ee) ---
# # # # # # # If you want to rotate the TARGET itself around Z (in the base frame), set theta_deg here.
# # # # # # theta_deg = 0
# # # # # # theta = np.deg2rad(theta_deg)
# # # # # # Rz_target = np.array(
# # # # # #     [
# # # # # #         [np.cos(theta), -np.sin(theta), 0.0, 0.0],
# # # # # #         [np.sin(theta),  np.cos(theta), 0.0, 0.0],
# # # # # #         [0.0,            0.0,           1.0, 0.0],
# # # # # #         [0.0,            0.0,           0.0, 1.0],
# # # # # #     ],
# # # # # #     dtype=float,
# # # # # # )

# # # # # # A = np.array(
# # # # # #     [
# # # # # #         [0, 0, -1, 1.7],
# # # # # #         [0, 1,  0, -0.4],
# # # # # #         [1, 0,  0, 0.1],
# # # # # #         [0, 0,  0, 1.0],
# # # # # #     ],
# # # # # #     dtype=float,
# # # # # # )
# # # # # # A = Rz_target @ A

# # # # # # # --- Search parameters (planar SE(2): dx, dy, yaw) ---
# # # # # # MAX_BASE_TOTAL_TRANS = 3.0   # max total translation budget (m)
# # # # # # MAX_BASE_TOTAL_YAW_DEG = 180.0  # max total yaw budget (deg), safety
# # # # # # MAX_STEP_TRANS = 0.30        # clamp per translation command (m)
# # # # # # MAX_STEP_YAW_DEG = 30.0      # clamp per yaw command (deg)

# # # # # # R_STEP = 0.05                # radial step for search (m)
# # # # # # R_MAX = 2.0                  # max search radius (m) (should be <= MAX_BASE_TOTAL_TRANS)
# # # # # # ANGLES = 16                  # samples per ring

# # # # # # # Try yaw in "small first" order; includes 180°.
# # # # # # YAW_LIST_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180]

# # # # # # ARM_DURATION = 2.0
# # # # # # BASE_TIMEOUT = 3.0

# # # # # # # If True: base.reset_odometry() is called before EACH goto_odom, meaning goto_odom is treated as an absolute
# # # # # # # motion in the freshly-reset odom frame (common pattern for Reachy base scripts).
# # # # # # RESET_ODOM_BEFORE_EACH_CMD = True
# # # # # # # ---------------------------------------


# # # # # # def se2_T(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # # # #     """4x4 SE(2) transform (odom <- base): translate (dx,dy) and rotate yaw about Z."""
# # # # # #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
# # # # # #     T = np.eye(4, dtype=float)
# # # # # #     T[0, 0] = c
# # # # # #     T[0, 1] = -s
# # # # # #     T[1, 0] = s
# # # # # #     T[1, 1] = c
# # # # # #     T[0, 3] = dx
# # # # # #     T[1, 3] = dy
# # # # # #     return T


# # # # # # def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # # # #     """
# # # # # #     Base moves by (dx,dy,yaw) in odom => the SAME world/odom target expressed in the NEW base frame:
# # # # # #         A_new = inv(T) @ A_old
# # # # # #     where T is the base motion (odom <- new_base) relative to old base/odom.
# # # # # #     """
# # # # # #     T = se2_T(dx, dy, yaw_rad)
# # # # # #     return np.linalg.inv(T) @ A_cur


# # # # # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # # # # #     """Return True if robot accepted the command (reachable)."""
# # # # # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # # # # #     gid = getattr(resp, "id", None)
# # # # # #     print("[ARM] goto id:", gid)
# # # # # #     return gid is not None and gid != -1


# # # # # # def base_goto_odom(base: BaseController, x: float, y: float, theta_deg: float) -> None:
# # # # # #     """Wrapper to optionally reset odometry before sending a command."""
# # # # # #     if RESET_ODOM_BEFORE_EACH_CMD:
# # # # # #         base.reset_odometry()
# # # # # #         time.sleep(0.05)

# # # # # #     base.goto_odom(
# # # # # #         x=float(x),
# # # # # #         y=float(y),
# # # # # #         theta=float(theta_deg),  # NOTE: BaseController commonly expects degrees; keep consistent with your scripts.
# # # # # #         wait=True,
# # # # # #         distance_tolerance=0.05,
# # # # # #         angle_tolerance=5.0,
# # # # # #         timeout=BASE_TIMEOUT,
# # # # # #     )


# # # # # # def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
# # # # # #     """Execute a yaw rotation (deg) in clamped steps. Returns total yaw executed (deg, abs-sum)."""
# # # # # #     remaining = float(yaw_deg_goal)
# # # # # #     total_abs = 0.0

# # # # # #     while abs(remaining) > 1e-3:
# # # # # #         step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
# # # # # #         print(f"[BASE] yaw step goto_odom(theta={step:.1f} deg)")
# # # # # #         base_goto_odom(base, x=0.0, y=0.0, theta_deg=step)

# # # # # #         total_abs += abs(step)
# # # # # #         if total_abs > MAX_BASE_TOTAL_YAW_DEG:
# # # # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

# # # # # #         remaining -= step

# # # # # #     return total_abs


# # # # # # def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
# # # # # #     """Execute translation in clamped steps. Returns total translation executed (m, abs-sum)."""
# # # # # #     remaining_dx = float(dx_goal)
# # # # # #     remaining_dy = float(dy_goal)
# # # # # #     total = 0.0

# # # # # #     while True:
# # # # # #         dist = float(np.hypot(remaining_dx, remaining_dy))
# # # # # #         if dist < 1e-6:
# # # # # #             break

# # # # # #         s = 1.0
# # # # # #         if dist > MAX_STEP_TRANS:
# # # # # #             s = MAX_STEP_TRANS / dist

# # # # # #         sx = remaining_dx * s
# # # # # #         sy = remaining_dy * s

# # # # # #         print(f"[BASE] translation step goto_odom(x={sx:.3f}, y={sy:.3f})")
# # # # # #         base_goto_odom(base, x=sx, y=sy, theta_deg=0.0)

# # # # # #         step_dist = float(np.hypot(sx, sy))
# # # # # #         total += step_dist
# # # # # #         if total > MAX_BASE_TOTAL_TRANS:
# # # # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

# # # # # #         remaining_dx -= sx
# # # # # #         remaining_dy -= sy

# # # # # #     return total


# # # # # # def main() -> int:
# # # # # #     # --- connect ---
# # # # # #     cfg = ReachyConfig(host=HOST)
# # # # # #     client = ReachyClient(cfg)
# # # # # #     client.connect()
# # # # # #     client.turn_on_all()

# # # # # #     arm = client._get_arm(SIDE)
# # # # # #     base = BaseController(client=client, world=None)

# # # # # #     print("[BASE] Resetting odometry...")
# # # # # #     base.reset_odometry()
# # # # # #     time.sleep(0.5)

# # # # # #     A_cur = A.copy()

# # # # # #     print("\n[1] First try without moving base")
# # # # # #     if try_arm(arm, A_cur):
# # # # # #         print("[SUCCESS] Reached without base move.")
# # # # # #         # arm.turn_off_smoothly()
# # # # # #         try:
# # # # # #             reachy = client.connect_reachy
# # # # # #             reachy.mobile_base.turn_off()
# # # # # #         except Exception:
# # # # # #             pass
# # # # # #         print("Done.")
# # # # # #         return 0

# # # # # #     print("\n[2] Unreachable -> SE(2) search over (dx,dy,yaw)")
# # # # # #     found = False
# # # # # #     best = None

# # # # # #     radii = np.arange(R_STEP, R_MAX + 1e-9, R_STEP)

# # # # # #     for yaw_deg in YAW_LIST_DEG:
# # # # # #         yaw = float(np.deg2rad(yaw_deg))

# # # # # #         for r in radii:
# # # # # #             for i in range(ANGLES):
# # # # # #                 ang = 2.0 * np.pi * (i / ANGLES)
# # # # # #                 dx = float(r * np.cos(ang))
# # # # # #                 dy = float(r * np.sin(ang))

# # # # # #                 # Respect global translation travel budget for a single-shot plan
# # # # # #                 if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS:
# # # # # #                     continue

# # # # # #                 A_try = apply_base_se2(A_cur, dx, dy, yaw)

# # # # # #                 print(
# # # # # #                     f"\n[SEARCH] yaw={yaw_deg:>4.0f}deg r={r:.2f} -> "
# # # # # #                     f"candidate base move dx={dx:.3f}, dy={dy:.3f}"
# # # # # #                 )

# # # # # #                 # Try arm WITHOUT moving base yet (cheap feasibility test)
# # # # # #                 if try_arm(arm, A_try):
# # # # # #                     print("[SEARCH] Found a feasible SE(2) offset.")
# # # # # #                     found = True
# # # # # #                     best = (dx, dy, yaw_deg)
# # # # # #                     break

# # # # # #             if found:
# # # # # #                 break
# # # # # #         if found:
# # # # # #             break

# # # # # #     if not found or best is None:
# # # # # #         print("[FAIL] No feasible base SE(2) offset found within search region.")
# # # # # #         # arm.turn_off_smoothly()
# # # # # #         try:
# # # # # #             reachy = client.connect_reachy
# # # # # #             reachy.mobile_base.turn_off()
# # # # # #         except Exception:
# # # # # #             pass
# # # # # #         print("Done.")
# # # # # #         return 1

# # # # # #     dx_goal, dy_goal, yaw_deg_goal = best
# # # # # #     print(
# # # # # #         f"\n[BASE] Best motion: dx={dx_goal:.3f}, dy={dy_goal:.3f}, yaw={yaw_deg_goal:.0f}deg "
# # # # # #         f"(will be split into steps)"
# # # # # #     )

# # # # # #     # --- Execute base motion ---
# # # # # #     try:
# # # # # #         # Often safer to rotate first, then translate.
# # # # # #         yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# # # # # #         trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# # # # # #         print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")
# # # # # #     except Exception as e:
# # # # # #         print("[FAIL] Base execution failed:", e)
# # # # # #         # arm.turn_off_smoothly()
# # # # # #         try:
# # # # # #             reachy = client.connect_reachy
# # # # # #             reachy.mobile_base.turn_off()
# # # # # #         except Exception:
# # # # # #             pass
# # # # # #         return 1

# # # # # #     # Update target in the new base frame using the SAME transform we executed
# # # # # #     A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, np.deg2rad(yaw_deg_goal))

# # # # # #     print("\n[3] Retrying arm.goto after base move")
# # # # # #     if try_arm(arm, A_cur):
# # # # # #         print("[SUCCESS] Reached after base assist with yaw.")
# # # # # #         code = 0
# # # # # #     else:
# # # # # #         print("[FAIL] Was feasible in search but failed after executing base motion (odom drift / slip?).")
# # # # # #         code = 1

# # # # # #     arm.turn_off_smoothly()
# # # # # #     try:
# # # # # #         reachy = client.connect_reachy
# # # # # #         reachy.mobile_base.turn_off()
# # # # # #     except Exception:
# # # # # #         pass

# # # # # #     print("Done.")
# # # # # #     return code


# # # # # # if __name__ == "__main__":
# # # # # #     raise SystemExit(main())


# # # # # # import time
# # # # # # import numpy as np

# # # # # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # # # # from reachy2_stack.core.client import ReachyClient
# # # # # # from reachy2_stack.control.base import BaseController

# # # # # # # ---------------- CONFIG ----------------
# # # # # # HOST = "192.168.1.71"
# # # # # # SIDE = "left"

# # # # # # # Target pose (base <- ee)
# # # # # # theta = np.deg2rad(0.0)
# # # # # # Rz = np.array([
# # # # # #     [np.cos(theta), -np.sin(theta), 0.0, 0.0],
# # # # # #     [np.sin(theta),  np.cos(theta), 0.0, 0.0],
# # # # # #     [0.0,            0.0,           1.0, 0.0],
# # # # # #     [0.0,            0.0,           0.0, 1.0],
# # # # # # ])
# # # # # # A = np.array(
# # # # # #     [
# # # # # #         [0, 0, -1, 0.1],
# # # # # #         [0, 1,  0, 0.4],
# # # # # #         [1, 0,  0, 0.1],
# # # # # #         [0, 0,  0, 1.0],
# # # # # #     ],
# # # # # #     dtype=float,
# # # # # # )
# # # # # # A = Rz @ A


# # # # # # # Search parameters (planar)
# # # # # # MAX_BASE_TOTAL = 3     # max distance base is allowed to travel in total (m)
# # # # # # MAX_STEP = 0.30           # single base command clamp (m)  (your base.goto_odom limit comfort)
# # # # # # R_STEP = 0.05             # radial step for search (m)
# # # # # # R_MAX = 2.0               # max search radius (m) (must be <= MAX_BASE_TOTAL)
# # # # # # ANGLES = 16               # samples per ring (increase to 24/32 if needed)

# # # # # # ARM_DURATION = 2.0
# # # # # # BASE_TIMEOUT = 3.0
# # # # # # # ---------------------------------------

# # # # # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # # # # #     """Return True if robot accepted the command (reachable)."""
# # # # # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # # # # #     gid = getattr(resp, "id", None)
# # # # # #     print("[ARM] goto id:", gid)
# # # # # #     return gid is not None and gid != -1

# # # # # # def apply_base_and_update(A_cur: np.ndarray, dx: float, dy: float) -> np.ndarray:
# # # # # #     """Base moves +dx,+dy in odom => target in new base frame shifts by -dx,-dy."""
# # # # # #     A2 = A_cur.copy()
# # # # # #     A2[0, 3] -= dx
# # # # # #     A2[1, 3] -= dy
# # # # # #     return A2

# # # # # # # --- connect ---
# # # # # # cfg = ReachyConfig(host=HOST)
# # # # # # client = ReachyClient(cfg)
# # # # # # client.connect()
# # # # # # client.turn_on_all()

# # # # # # arm = client._get_arm(SIDE)
# # # # # # base = BaseController(client=client, world=None)


# # # # # # print("[BASE] Resetting odometry...")
# # # # # # base.reset_odometry()
# # # # # # time.sleep(0.5)

# # # # # # A_cur = A.copy()

# # # # # # print("\n[1] First try without moving base")
# # # # # # if try_arm(arm, A_cur):
# # # # # #     print("[SUCCESS] Reached without base move.")
# # # # # #     arm.turn_off_smoothly()
# # # # # #     print("Done.")
# # # # # #     raise SystemExit(0)

# # # # # # print("\n[2] Unreachable -> planar search for a base (dx,dy) that makes it reachable")
# # # # # # total_moved = 0.0

# # # # # # # Spiral / ring search: r = R_STEP .. R_MAX
# # # # # # radii = np.arange(R_STEP, R_MAX + 1e-9, R_STEP)

# # # # # # found = False
# # # # # # best_dxdy = None

# # # # # # for r in radii:
# # # # # #     # sample angles uniformly
# # # # # #     for i in range(ANGLES):
# # # # # #         theta = 2.0 * np.pi * (i / ANGLES)
# # # # # #         dx = float(r * np.cos(theta))
# # # # # #         dy = float(r * np.sin(theta))

# # # # # #         # Respect global travel budget
# # # # # #         if np.hypot(dx, dy) > MAX_BASE_TOTAL:
# # # # # #             continue

# # # # # #         # Candidate pose if base moved (dx,dy) from current base frame
# # # # # #         A_try = apply_base_and_update(A_cur, dx, dy)

# # # # # #         print(f"\n[SEARCH] r={r:.2f} theta={theta:.2f} -> candidate base move dx={dx:.3f}, dy={dy:.3f}")
# # # # # #         # Try arm WITHOUT moving base yet (cheap test by asking robot)
# # # # # #         if try_arm(arm, A_try):
# # # # # #             print("[SEARCH] Found a feasible base offset.")
# # # # # #             found = True
# # # # # #             best_dxdy = (dx, dy)
# # # # # #             break

# # # # # #     if found:
# # # # # #         break

# # # # # # if not found:
# # # # # #     print("[FAIL] No feasible base offset found within search radius.")
# # # # # #     arm.turn_off_smoothly()
# # # # # #     print("Done.")
# # # # # #     raise SystemExit(1)

# # # # # # dx, dy = best_dxdy

# # # # # # # Execute base move in possibly multiple clamped steps
# # # # # # remaining_dx = dx
# # # # # # remaining_dy = dy

# # # # # # print(f"\n[BASE] Executing base move to dx={dx:.3f}, dy={dy:.3f} (may be split into steps)")
# # # # # # while True:
# # # # # #     step = float(np.hypot(remaining_dx, remaining_dy))
# # # # # #     if step < 1e-6:
# # # # # #         break

# # # # # #     s = 1.0
# # # # # #     if step > MAX_STEP:
# # # # # #         s = MAX_STEP / step

# # # # # #     sx = remaining_dx * s
# # # # # #     sy = remaining_dy * s

# # # # # #     print(f"[BASE] step goto_odom(x={sx:.3f}, y={sy:.3f})")
# # # # # #     base.reset_odometry()  # you want absolute moves
# # # # # #     base.goto_odom(
# # # # # #         x=float(sx),
# # # # # #         y=float(sy),
# # # # # #         theta=0.0,
# # # # # #         wait=True,
# # # # # #         distance_tolerance=0.05,
# # # # # #         angle_tolerance=5.0,
# # # # # #         timeout=BASE_TIMEOUT,
# # # # # #     )
# # # # # #     total_moved += float(np.hypot(sx, sy))
# # # # # #     if total_moved > MAX_BASE_TOTAL:
# # # # # #         print("[FAIL] Exceeded MAX_BASE_TOTAL while moving base.")
# # # # # #         arm.turn_off_smoothly()
# # # # # #         raise SystemExit(1)

# # # # # #     # update target in new base frame
# # # # # #     A_cur = apply_base_and_update(A_cur, sx, sy)

# # # # # #     remaining_dx -= sx
# # # # # #     remaining_dy -= sy

# # # # # # print("\n[3] Retrying arm.goto after base move")
# # # # # # if try_arm(arm, A_cur):
# # # # # #     print("[SUCCESS] Reached after base assist.")
# # # # # # else:
# # # # # #     print("[FAIL] Unexpected: was feasible in search, but failed after moving base (odom drift/slip?).")

# # # # # # arm.turn_off_smoothly()
# # # # # # reachy = client.connect_reachy
# # # # # # reachy.mobile_base.turn_off()
# # # # # # print("Done.")


# # # # # #!/usr/bin/env python3
# # # # # import time
# # # # # import numpy as np

# # # # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # # # from reachy2_stack.core.client import ReachyClient
# # # # # from reachy2_stack.control.base import BaseController

# # # # # # ---------------- CONFIG ----------------
# # # # # HOST = "192.168.1.71"
# # # # # SIDE = "right"

# # # # # # --- Target pose (base <- ee) ---
# # # # # # If you want to rotate the TARGET itself around Z (in the base frame), set theta_deg here.
# # # # # theta_deg = 45.0
# # # # # theta = np.deg2rad(theta_deg)
# # # # # Rz_target = np.array(
# # # # #     [
# # # # #         [np.cos(theta), -np.sin(theta), 0.0, 0.0],
# # # # #         [np.sin(theta),  np.cos(theta), 0.0, 0.0],
# # # # #         [0.0,            0.0,           1.0, 0.0],
# # # # #         [0.0,            0.0,           0.0, 1.0],
# # # # #     ],
# # # # #     dtype=float,
# # # # # )

# # # # # A = np.array(
# # # # #     [
# # # # #         [0, 0, -1, 0.7],
# # # # #         [0, 1,  0, -0.4],
# # # # #         [1, 0,  0, 0.1],
# # # # #         [0, 0,  0, 1.0],
# # # # #     ],
# # # # #     dtype=float,
# # # # # )
# # # # # A = Rz_target @ A

# # # # # # --- Search parameters (planar SE(2): dx, dy, yaw) ---
# # # # # MAX_BASE_TOTAL_TRANS = 3.0   # max total translation budget (m)
# # # # # MAX_BASE_TOTAL_YAW_DEG = 180.0  # max total yaw budget (deg), safety
# # # # # MAX_STEP_TRANS = 0.30        # clamp per translation command (m)
# # # # # MAX_STEP_YAW_DEG = 30.0      # clamp per yaw command (deg)

# # # # # R_STEP = 0.05                # radial step for search (m)
# # # # # R_MAX = 2.0                  # max search radius (m) (should be <= MAX_BASE_TOTAL_TRANS)
# # # # # ANGLES = 16                  # samples per ring

# # # # # # Try yaw in "small first" order; includes 180°.
# # # # # YAW_LIST_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180]

# # # # # ARM_DURATION = 2.0
# # # # # BASE_TIMEOUT = 3.0

# # # # # # If True: base.reset_odometry() is called before EACH goto_odom, meaning goto_odom is treated as an absolute
# # # # # # motion in the freshly-reset odom frame (common pattern for Reachy base scripts).
# # # # # RESET_ODOM_BEFORE_EACH_CMD = True
# # # # # # ---------------------------------------


# # # # # def se2_T(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # # #     """4x4 SE(2) transform (odom <- base): translate (dx,dy) and rotate yaw about Z."""
# # # # #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
# # # # #     T = np.eye(4, dtype=float)
# # # # #     T[0, 0] = c
# # # # #     T[0, 1] = -s
# # # # #     T[1, 0] = s
# # # # #     T[1, 1] = c
# # # # #     T[0, 3] = dx
# # # # #     T[1, 3] = dy
# # # # #     return T


# # # # # def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # # #     """
# # # # #     Base moves by (dx,dy,yaw) in odom => the SAME world/odom target expressed in the NEW base frame:
# # # # #         A_new = inv(T) @ A_old
# # # # #     where T is the base motion (odom <- new_base) relative to old base/odom.
# # # # #     """
# # # # #     T = se2_T(dx, dy, yaw_rad)
# # # # #     return np.linalg.inv(T) @ A_cur


# # # # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # # # #     """Return True if robot accepted the command (reachable)."""
# # # # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # # # #     gid = getattr(resp, "id", None)
# # # # #     print("[ARM] goto id:", gid)
# # # # #     return gid is not None and gid != -1


# # # # # def base_goto_odom(base: BaseController, x: float, y: float, theta_deg: float) -> None:
# # # # #     """Wrapper to optionally reset odometry before sending a command."""
# # # # #     if RESET_ODOM_BEFORE_EACH_CMD:
# # # # #         base.reset_odometry()
# # # # #         time.sleep(0.05)

# # # # #     base.goto_odom(
# # # # #         x=float(x),
# # # # #         y=float(y),
# # # # #         theta=float(theta_deg),  # NOTE: BaseController commonly expects degrees; keep consistent with your scripts.
# # # # #         wait=True,
# # # # #         distance_tolerance=0.05,
# # # # #         angle_tolerance=5.0,
# # # # #         timeout=BASE_TIMEOUT,
# # # # #     )


# # # # # def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
# # # # #     """Execute yaw rotation in clamped steps using rotate_by. Returns total abs yaw executed (deg)."""
# # # # #     remaining = float(yaw_deg_goal)
# # # # #     total_abs = 0.0

# # # # #     while abs(remaining) > 1e-3:
# # # # #         step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
# # # # #         print(f"[BASE] rotate_by(theta={step:.1f} deg)")
# # # # #         g = base.rotate_by(theta=step, wait=True, degrees=True)
# # # # #         total_abs += abs(step)

# # # # #         if total_abs > MAX_BASE_TOTAL_YAW_DEG:
# # # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

# # # # #         remaining -= step

# # # # #     return total_abs


# # # # # def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
# # # # #     """Execute translation in clamped steps using translate_by. Returns total translation executed (m)."""
# # # # #     remaining_dx = float(dx_goal)
# # # # #     remaining_dy = float(dy_goal)
# # # # #     total = 0.0

# # # # #     while True:
# # # # #         dist = float(np.hypot(remaining_dx, remaining_dy))
# # # # #         if dist < 1e-6:
# # # # #             break

# # # # #         s = 1.0
# # # # #         if dist > MAX_STEP_TRANS:
# # # # #             s = MAX_STEP_TRANS / dist

# # # # #         sx = remaining_dx * s
# # # # #         sy = remaining_dy * s

# # # # #         print(f"[BASE] translate_by(x={sx:.3f}, y={sy:.3f})")
# # # # #         g = base.translate_by(x=float(sx), y=float(sy), wait=True)

# # # # #         step_dist = float(np.hypot(sx, sy))
# # # # #         total += step_dist
# # # # #         if total > MAX_BASE_TOTAL_TRANS:
# # # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

# # # # #         remaining_dx -= sx
# # # # #         remaining_dy -= sy

# # # # #     return total



# # # # # def main() -> int:
# # # # #     # --- connect ---
# # # # #     cfg = ReachyConfig(host=HOST)
# # # # #     client = ReachyClient(cfg)
# # # # #     client.connect()
# # # # #     client.turn_on_all()

# # # # #     arm = client._get_arm(SIDE)
# # # # #     base = BaseController(client=client, world=None)

# # # # #     print("[BASE] Resetting odometry...")
# # # # #     base.reset_odometry()
# # # # #     time.sleep(0.5)

# # # # #     A_cur = A.copy()

# # # # #     print("\n[1] First try without moving base")
# # # # #     if try_arm(arm, A_cur):
# # # # #         print("[SUCCESS] Reached without base move.")
# # # # #         # arm.turn_off_smoothly()
# # # # #         try:
# # # # #             reachy = client.connect_reachy
# # # # #             reachy.mobile_base.turn_off()
# # # # #         except Exception:
# # # # #             pass
# # # # #         print("Done.")
# # # # #         return 0

# # # # #     print("\n[2] Unreachable -> SE(2) search over (dx,dy,yaw)")
# # # # #     found = False
# # # # #     best = None

# # # # #     radii = np.arange(R_STEP, R_MAX + 1e-9, R_STEP)

# # # # #     for yaw_deg in YAW_LIST_DEG:
# # # # #         yaw = float(np.deg2rad(yaw_deg))

# # # # #         for r in radii:
# # # # #             for i in range(ANGLES):
# # # # #                 ang = 2.0 * np.pi * (i / ANGLES)
# # # # #                 dx = float(r * np.cos(ang))
# # # # #                 dy = float(r * np.sin(ang))

# # # # #                 # Respect global translation travel budget for a single-shot plan
# # # # #                 if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS:
# # # # #                     continue

# # # # #                 A_try = apply_base_se2(A_cur, dx, dy, yaw)

# # # # #                 print(
# # # # #                     f"\n[SEARCH] yaw={yaw_deg:>4.0f}deg r={r:.2f} -> "
# # # # #                     f"candidate base move dx={dx:.3f}, dy={dy:.3f}"
# # # # #                 )

# # # # #                 # Try arm WITHOUT moving base yet (cheap feasibility test)
# # # # #                 if try_arm(arm, A_try):
# # # # #                     print("[SEARCH] Found a feasible SE(2) offset.")
# # # # #                     found = True
# # # # #                     best = (dx, dy, yaw_deg)
# # # # #                     break

# # # # #             if found:
# # # # #                 break
# # # # #         if found:
# # # # #             break

# # # # #     if not found or best is None:
# # # # #         print("[FAIL] No feasible base SE(2) offset found within search region.")
# # # # #         # arm.turn_off_smoothly()
# # # # #         try:
# # # # #             reachy = client.connect_reachy
# # # # #             reachy.mobile_base.turn_off()
# # # # #         except Exception:
# # # # #             pass
# # # # #         print("Done.")
# # # # #         return 1

# # # # #     dx_goal, dy_goal, yaw_deg_goal = best
# # # # #     print(
# # # # #         f"\n[BASE] Best motion: dx={dx_goal:.3f}, dy={dy_goal:.3f}, yaw={yaw_deg_goal:.0f}deg "
# # # # #         f"(will be split into steps)"
# # # # #     )

# # # # #     # --- Execute base motion ---
# # # # #     try:
# # # # #         # Often safer to rotate first, then translate.
# # # # #         yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# # # # #         trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# # # # #         print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")
# # # # #     except Exception as e:
# # # # #         print("[FAIL] Base execution failed:", e)
# # # # #         # arm.turn_off_smoothly()
# # # # #         try:
# # # # #             reachy = client.connect_reachy
# # # # #             reachy.mobile_base.turn_off()
# # # # #         except Exception:
# # # # #             pass
# # # # #         return 1

# # # # #     # Update target in the new base frame using the SAME transform we executed
# # # # #     A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, np.deg2rad(yaw_deg_goal))

# # # # #     print("\n[3] Retrying arm.goto after base move")
# # # # #     if try_arm(arm, A_cur):
# # # # #         print("[SUCCESS] Reached after base assist with yaw.")
# # # # #         code = 0
# # # # #     else:
# # # # #         print("[FAIL] Was feasible in search but failed after executing base motion (odom drift / slip?).")
# # # # #         code = 1

# # # # #     arm.turn_off_smoothly()
# # # # #     try:
# # # # #         reachy = client.connect_reachy
# # # # #         reachy.mobile_base.turn_off()
# # # # #     except Exception:
# # # # #         pass

# # # # #     print("Done.")
# # # # #     return code


# # # # # if __name__ == "__main__":
# # # # #     raise SystemExit(main())


# # # # #!/usr/bin/env python3
# # # # from __future__ import annotations

# # # # import time
# # # # import numpy as np

# # # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # # from reachy2_stack.core.client import ReachyClient
# # # # from reachy2_stack.control.base import BaseController

# # # # # ---------------- CONFIG ----------------
# # # # HOST = "192.168.1.71"
# # # # SIDE = "right"

# # # # # --- Target pose (base <- ee) ---
# # # # # Rotate the TARGET itself around Z in the base frame (optional)
# # # # TARGET_YAW_DEG = 0.0

# # # # A = np.array(
# # # #     [
# # # #         [0, 0, -1, -0.7],
# # # #         [0, 1,  0, -0.4],
# # # #         [1, 0,  0, 0.1],
# # # #         [0, 0,  0, 1.0],
# # # #     ],
# # # #     dtype=float,
# # # # )

# # # # # --- Search / safety budgets ---
# # # # MAX_BASE_TOTAL_TRANS = 3.0        # max total translation budget (m)
# # # # MAX_BASE_TOTAL_YAW_DEG = 180.0    # max total yaw budget (deg)

# # # # MAX_STEP_TRANS = 0.30             # clamp per translation command (m)
# # # # MAX_STEP_YAW_DEG = 30.0           # clamp per yaw command (deg)

# # # # # Heuristic tries (fast)
# # # # LINE_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
# # # # YAW_OFFSETS_DEG = [0.0, 20.0, -20.0, 45.0, -45.0, 90.0, -90.0, 180.0]  # used around yaw_guess

# # # # # Fallback coarse ring search (still much smaller than your original)
# # # # R_STEP_COARSE = 0.20
# # # # R_MAX = 2.0
# # # # ANGLE_PRIORITY_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]

# # # # # Arm command parameters
# # # # ARM_DURATION = 2.0
# # # # # ---------------------------------------


# # # # def wrap180(deg: float) -> float:
# # # #     """Wrap degrees to (-180, 180]."""
# # # #     return (deg + 180.0) % 360.0 - 180.0


# # # # def Rz(deg: float) -> np.ndarray:
# # # #     th = np.deg2rad(deg)
# # # #     c, s = float(np.cos(th)), float(np.sin(th))
# # # #     R = np.eye(4, dtype=float)
# # # #     R[0, 0] = c
# # # #     R[0, 1] = -s
# # # #     R[1, 0] = s
# # # #     R[1, 1] = c
# # # #     return R


# # # # def se2_T(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # #     """4x4 SE(2) transform (odom <- base): translate (dx,dy) and rotate yaw about Z."""
# # # #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
# # # #     T = np.eye(4, dtype=float)
# # # #     T[0, 0] = c
# # # #     T[0, 1] = -s
# # # #     T[1, 0] = s
# # # #     T[1, 1] = c
# # # #     T[0, 3] = dx
# # # #     T[1, 3] = dy
# # # #     return T


# # # # def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # # #     """
# # # #     If the base undergoes relative motion T (translate dx,dy + rotate yaw) in the world,
# # # #     the same world target expressed in the NEW base frame is:
# # # #         A_new = inv(T) @ A_old
# # # #     """
# # # #     T = se2_T(dx, dy, yaw_rad)
# # # #     return np.linalg.inv(T) @ A_cur


# # # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # # #     """Return True if robot accepted the command (reachable)."""
# # # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # # #     gid = getattr(resp, "id", None)
# # # #     # Many stacks use None/-1 to signal failure
# # # #     ok = gid is not None and gid != -1
# # # #     print("[ARM] goto id:", gid, "=>", "OK" if ok else "NO")
# # # #     return ok


# # # # def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
# # # #     """Execute yaw rotation in clamped steps using rotate_by. Returns total abs yaw executed (deg)."""
# # # #     remaining = float(yaw_deg_goal)
# # # #     total_abs = 0.0

# # # #     while abs(remaining) > 1e-3:
# # # #         step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
# # # #         print(f"[BASE] rotate_by(theta={step:.1f} deg)")
# # # #         base.rotate_by(theta=step, wait=True, degrees=True)

# # # #         total_abs += abs(step)
# # # #         if total_abs > MAX_BASE_TOTAL_YAW_DEG:
# # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

# # # #         remaining -= step

# # # #     return total_abs


# # # # def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
# # # #     """Execute translation in clamped steps using translate_by. Returns total translation executed (m)."""
# # # #     remaining_dx = float(dx_goal)
# # # #     remaining_dy = float(dy_goal)
# # # #     total = 0.0

# # # #     while True:
# # # #         dist = float(np.hypot(remaining_dx, remaining_dy))
# # # #         if dist < 1e-6:
# # # #             break

# # # #         s = 1.0
# # # #         if dist > MAX_STEP_TRANS:
# # # #             s = MAX_STEP_TRANS / dist

# # # #         sx = remaining_dx * s
# # # #         sy = remaining_dy * s

# # # #         print(f"[BASE] translate_by(x={sx:.3f}, y={sy:.3f})")
# # # #         base.translate_by(x=float(sx), y=float(sy), wait=True)

# # # #         step_dist = float(np.hypot(sx, sy))
# # # #         total += step_dist
# # # #         if total > MAX_BASE_TOTAL_TRANS:
# # # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

# # # #         remaining_dx -= sx
# # # #         remaining_dy -= sy

# # # #     return total


# # # # def line_candidates_toward_target(A_cur: np.ndarray, steps: list[float]) -> list[tuple[float, float]]:
# # # #     """
# # # #     Generate (dx,dy) candidates moving base toward the target direction in the base frame,
# # # #     using increasing step sizes.
# # # #     """
# # # #     p = A_cur[:2, 3].astype(float)
# # # #     n = float(np.linalg.norm(p))
# # # #     if n < 1e-9:
# # # #         u = np.array([1.0, 0.0], dtype=float)
# # # #     else:
# # # #         u = p / n

# # # #     cands = []
# # # #     for s in steps:
# # # #         dx = float(u[0] * s)
# # # #         dy = float(u[1] * s)
# # # #         if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # # #             cands.append((dx, dy))

# # # #     # also try pure-x and pure-y (often useful if constraints are asymmetric)
# # # #     for s in steps[:6]:
# # # #         for (dx, dy) in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
# # # #             if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # # #                 cands.append((float(dx), float(dy)))

# # # #     # de-dup while preserving order
# # # #     seen = set()
# # # #     uniq = []
# # # #     for dx, dy in cands:
# # # #         key = (round(dx, 4), round(dy, 4))
# # # #         if key not in seen:
# # # #             seen.add(key)
# # # #             uniq.append((dx, dy))
# # # #     return uniq


# # # # def coarse_ring_candidates(r_step: float, r_max: float, angle_priority_deg: list[int]) -> list[tuple[float, float]]:
# # # #     """Generate coarse ring (dx,dy) candidates, front/diag first."""
# # # #     radii = np.arange(r_step, r_max + 1e-9, r_step)
# # # #     cands = []
# # # #     for r in radii:
# # # #         for ang_deg in angle_priority_deg:
# # # #             th = np.deg2rad(float(ang_deg))
# # # #             dx = float(r * np.cos(th))
# # # #             dy = float(r * np.sin(th))
# # # #             if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # # #                 cands.append((dx, dy))
# # # #     return cands


# # # # def find_base_assist(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
# # # #     """
# # # #     Returns best (dx, dy, yaw_deg) found, or None.
# # # #     Heuristic order:
# # # #       1) yaw=0, translate along target direction (few steps)
# # # #       2) yaw around yaw_guess, translate along target direction (few steps)
# # # #       3) coarse ring with a small yaw set
# # # #     """
# # # #     p = A_cur[:2, 3].astype(float)
# # # #     yaw_guess = wrap180(float(np.degrees(np.arctan2(p[1], p[0]))))  # points base toward target direction

# # # #     # Stage 1: yaw=0, line candidates
# # # #     print("\n[SEARCH-1] yaw=0, line candidates toward target")
# # # #     for dx, dy in line_candidates_toward_target(A_cur, LINE_STEP_TRIES):
# # # #         A_try = apply_base_se2(A_cur, dx, dy, 0.0)
# # # #         print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
# # # #         if try_arm(arm, A_try):
# # # #             return dx, dy, 0.0

# # # #     # Stage 2: yaw around guess, then line candidates
# # # #     print("\n[SEARCH-2] yaw around yaw_guess, then line candidates")
# # # #     yaw_list = [wrap180(yaw_guess + off) for off in YAW_OFFSETS_DEG]
# # # #     # de-dup and small-first order
# # # #     yaw_list = list(dict.fromkeys([float(y) for y in yaw_list]))

# # # #     for yaw_deg in yaw_list:
# # # #         yaw_rad = float(np.deg2rad(yaw_deg))
# # # #         # For yaw-only, also try (dx,dy) = (0,0) first
# # # #         print(f"\n[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
# # # #         if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad)):
# # # #             return 0.0, 0.0, yaw_deg

# # # #         for dx, dy in line_candidates_toward_target(A_cur, LINE_STEP_TRIES[:8]):
# # # #             A_try = apply_base_se2(A_cur, dx, dy, yaw_rad)
# # # #             print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
# # # #             if try_arm(arm, A_try):
# # # #                 return dx, dy, yaw_deg

# # # #     # Stage 3: coarse ring + small yaw set (still far fewer than full scan)
# # # #     print("\n[SEARCH-3] coarse rings + small yaw set (fallback)")
# # # #     yaw_fallback = [0.0, yaw_guess, wrap180(yaw_guess + 45.0), wrap180(yaw_guess - 45.0), 90.0, -90.0, 180.0]
# # # #     yaw_fallback = list(dict.fromkeys([float(wrap180(y)) for y in yaw_fallback]))

# # # #     ring_cands = coarse_ring_candidates(R_STEP_COARSE, R_MAX, ANGLE_PRIORITY_DEG)
# # # #     for yaw_deg in yaw_fallback:
# # # #         yaw_rad = float(np.deg2rad(yaw_deg))
# # # #         for dx, dy in ring_cands:
# # # #             print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
# # # #             A_try = apply_base_se2(A_cur, dx, dy, yaw_rad)
# # # #             if try_arm(arm, A_try):
# # # #                 return dx, dy, yaw_deg

# # # #     return None


# # # # def main() -> int:
# # # #     # Rotate target pose around Z if requested
# # # #     A_target = (Rz(TARGET_YAW_DEG) @ A).astype(float)

# # # #     cfg = ReachyConfig(host=HOST)
# # # #     client = ReachyClient(cfg)
# # # #     client.connect()
# # # #     client.turn_on_all()

# # # #     arm = client._get_arm(SIDE)
# # # #     base = BaseController(client=client, world=None)

# # # #     # IMPORTANT: we do NOT reset odometry repeatedly; we only do relative base motions.
# # # #     # You may still want a single reset at start if your experiment assumes zeroed odom.
# # # #     print("[BASE] (optional) Resetting odometry once at start...")
# # # #     base.reset_odometry()
# # # #     time.sleep(0.5)

# # # #     A_cur = A_target.copy()

# # # #     try:
# # # #         print("\n[1] First try without moving base")
# # # #         if try_arm(arm, A_cur):
# # # #             print("[SUCCESS] Reached without base move.")
# # # #             return 0

# # # #         print("\n[2] Unreachable -> heuristic base-assist search (fast ordering)")
# # # #         best = find_base_assist(arm, A_cur)

# # # #         if best is None:
# # # #             print("[FAIL] No feasible base offset found in heuristic search.")
# # # #             return 1

# # # #         dx_goal, dy_goal, yaw_deg_goal = best
# # # #         print(
# # # #             f"\n[BASE] Best motion found: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f}deg"
# # # #         )

# # # #         # Execute base motion (rotate then translate)
# # # #         # NOTE: This does NOT modify odom reference frame; it’s relative motion.
# # # #         yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# # # #         # Update target in base frame after yaw (match execution order)
# # # #         A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal))

# # # #         trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# # # #         # Update target in base frame after translation
# # # #         A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0)

# # # #         print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")

# # # #         print("\n[3] Retrying arm.goto after base assist")
# # # #         # if try_arm(arm, A_cur):
# # # #         #     print("[SUCCESS] Reached after base assist.")
# # # #         #     return 0
# # # #         # else:
# # # #         #     print("[FAIL] Found feasible in search but failed after executing base motion (drift/slip/frames).")
# # # #         #     return 1

# # # #     finally:
# # # #         try:
# # # #             arm.turn_off_smoothly()
# # # #         except Exception:
# # # #             pass
# # # #         try:
# # # #             reachy = client.connect_reachy
# # # #             reachy.mobile_base.turn_off()
# # # #         except Exception:
# # # #             pass
# # # #         client.close()
# # # #         print("Done.")


# # # # if __name__ == "__main__":
# # # #     raise SystemExit(main())


# # # #!/usr/bin/env python3
# # # from __future__ import annotations

# # # import time
# # # import numpy as np

# # # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # # from reachy2_stack.core.client import ReachyClient
# # # from reachy2_stack.control.base import BaseController

# # # # ---------------- CONFIG ----------------
# # # HOST = "192.168.1.71"
# # # SIDE = "right"

# # # # Rotate the TARGET itself around Z in the base frame (optional)
# # # TARGET_YAW_DEG = 0.0

# # # # Example target (base <- ee)
# # # A = np.array(
# # #     [
# # #         [0, 0, -1, 1.4],
# # #         [0, 1,  0, -0.4],
# # #         [1, 0,  0,  0.1],
# # #         [0, 0,  0,  1.0],
# # #     ],
# # #     dtype=float,
# # # )

# # # # --- Safety / budgets ---
# # # MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
# # # MAX_BASE_TOTAL_YAW_DEG = 180.0    # max total yaw (deg)

# # # MAX_STEP_TRANS = 0.30             # clamp per translate_by (m)
# # # MAX_STEP_YAW_DEG = 30.0           # clamp per rotate_by (deg)

# # # # Arm command parameters
# # # ARM_DURATION = 2.0

# # # # Heuristic ordering params
# # # LINE_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]

# # # # Extra translation tries that often help (axis + diagonals)
# # # AXIS_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
# # # DIAG_FACTOR = 0.7  # for diag tries: (±s, ±0.7s)

# # # # Rotation is LAST resort: small first, no 180 until the very end
# # # YAW_SMALL_DEG = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0]
# # # YAW_FALLBACK_DEG = [90.0, -90.0, 180.0]  # only if everything fails

# # # # Fallback coarse ring search (translation-only first)
# # # R_STEP_COARSE = 0.20
# # # R_MAX = 2.0
# # # ANGLE_PRIORITY_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]
# # # # ---------------------------------------


# # # def wrap180(deg: float) -> float:
# # #     return (deg + 180.0) % 360.0 - 180.0


# # # def Rz(deg: float) -> np.ndarray:
# # #     th = np.deg2rad(float(deg))
# # #     c, s = float(np.cos(th)), float(np.sin(th))
# # #     R = np.eye(4, dtype=float)
# # #     R[0, 0] = c
# # #     R[0, 1] = -s
# # #     R[1, 0] = s
# # #     R[1, 1] = c
# # #     return R


# # # def se2_T_translate_then_rotate(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # #     """
# # #     IMPORTANT: algorithm prioritizes translation before rotation.
# # #     We model the executed motion as:
# # #         translate_by(dx,dy) THEN rotate_by(yaw)
# # #     so:
# # #         T = Trans(dx,dy) @ R(yaw)
# # #     """
# # #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

# # #     Tt = np.eye(4, dtype=float)
# # #     Tt[0, 3] = float(dx)
# # #     Tt[1, 3] = float(dy)

# # #     R = np.eye(4, dtype=float)
# # #     R[0, 0] = c
# # #     R[0, 1] = -s
# # #     R[1, 0] = s
# # #     R[1, 1] = c

# # #     return Tt @ R


# # # def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# # #     """
# # #     Base motion T is applied in the world/odom; update target in the NEW base frame:
# # #         A_new = inv(T) @ A_old
# # #     """
# # #     T = se2_T_translate_then_rotate(dx, dy, yaw_rad)
# # #     return np.linalg.inv(T) @ A_cur


# # # def try_arm(arm, A_try: np.ndarray) -> bool:
# # #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# # #     gid = getattr(resp, "id", None)
# # #     ok = gid is not None and gid != -1
# # #     print("[ARM] goto id:", gid, "=>", "OK" if ok else "NO")
# # #     return ok


# # # def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
# # #     remaining_dx = float(dx_goal)
# # #     remaining_dy = float(dy_goal)
# # #     total = 0.0

# # #     while True:
# # #         dist = float(np.hypot(remaining_dx, remaining_dy))
# # #         if dist < 1e-6:
# # #             break

# # #         s = 1.0
# # #         if dist > MAX_STEP_TRANS:
# # #             s = MAX_STEP_TRANS / dist

# # #         sx = remaining_dx * s
# # #         sy = remaining_dy * s

# # #         print(f"[BASE] translate_by(x={sx:+.3f}, y={sy:+.3f})")
# # #         base.translate_by(x=float(sx), y=float(sy), wait=True)

# # #         step_dist = float(np.hypot(sx, sy))
# # #         total += step_dist
# # #         if total > MAX_BASE_TOTAL_TRANS:
# # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

# # #         remaining_dx -= sx
# # #         remaining_dy -= sy

# # #     return total


# # # def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
# # #     remaining = float(yaw_deg_goal)
# # #     total_abs = 0.0

# # #     while abs(remaining) > 1e-3:
# # #         step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
# # #         print(f"[BASE] rotate_by(theta={step:.1f} deg)")
# # #         base.rotate_by(theta=step, wait=True, degrees=True)

# # #         total_abs += abs(step)
# # #         if total_abs > MAX_BASE_TOTAL_YAW_DEG:
# # #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

# # #         remaining -= step

# # #     return total_abs


# # # def translation_priority_candidates(A_cur: np.ndarray) -> list[tuple[float, float]]:
# # #     """
# # #     Candidates ordered to prioritize translation:
# # #       1) Move along direction of target translation p = [x,y] (small->big)
# # #       2) Axis moves ±x, ±y
# # #       3) Diagonals
# # #     """
# # #     p = A_cur[:2, 3].astype(float)
# # #     n = float(np.linalg.norm(p))
# # #     if n < 1e-9:
# # #         u = np.array([1.0, 0.0], dtype=float)
# # #     else:
# # #         u = p / n

# # #     cands: list[tuple[float, float]] = []

# # #     # 1) along target direction (toward p)
# # #     for s in LINE_STEP_TRIES:
# # #         dx = float(u[0] * s)
# # #         dy = float(u[1] * s)
# # #         if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # #             cands.append((dx, dy))

# # #     # 1b) also try opposite direction (sometimes you need to move away due to elbow limits)
# # #     for s in LINE_STEP_TRIES[:6]:
# # #         dx = float(-u[0] * s)
# # #         dy = float(-u[1] * s)
# # #         if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # #             cands.append((dx, dy))

# # #     # 2) axis moves
# # #     for s in AXIS_STEP_TRIES:
# # #         for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
# # #             cands.append((float(dx), float(dy)))

# # #     # 3) diagonals
# # #     for s in AXIS_STEP_TRIES:
# # #         a = float(s)
# # #         b = float(DIAG_FACTOR * s)
# # #         for dx, dy in [(a, b), (a, -b), (-a, b), (-a, -b)]:
# # #             cands.append((dx, dy))

# # #     # filter + dedup (preserve order)
# # #     seen = set()
# # #     uniq: list[tuple[float, float]] = []
# # #     for dx, dy in cands:
# # #         if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
# # #             continue
# # #         key = (round(dx, 4), round(dy, 4))
# # #         if key not in seen:
# # #             seen.add(key)
# # #             uniq.append((dx, dy))
# # #     return uniq


# # # def coarse_ring_candidates(r_step: float, r_max: float, angle_priority_deg: list[int]) -> list[tuple[float, float]]:
# # #     radii = np.arange(r_step, r_max + 1e-9, r_step)
# # #     cands: list[tuple[float, float]] = []
# # #     for r in radii:
# # #         for ang_deg in angle_priority_deg:
# # #             th = np.deg2rad(float(ang_deg))
# # #             dx = float(r * np.cos(th))
# # #             dy = float(r * np.sin(th))
# # #             if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# # #                 cands.append((dx, dy))
# # #     return cands


# # # def find_base_assist_translation_first(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
# # #     """
# # #     STRICT priority:
# # #       A) translation-only search (yaw=0) with strong candidate ordering
# # #       B) translation-only coarse rings (yaw=0)
# # #       C) only then: yaw search (small yaws), optionally with tiny translation
# # #       D) final: 90/-90/180 yaw fallback
# # #     """
# # #     # A) translation-only, yaw=0
# # #     print("\n[SEARCH-A] translation-only (yaw=0) prioritized")
# # #     for dx, dy in translation_priority_candidates(A_cur):
# # #         print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
# # #         if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0)):
# # #             return dx, dy, 0.0

# # #     # B) coarse rings translation-only
# # #     print("\n[SEARCH-B] translation-only coarse rings (yaw=0)")
# # #     for dx, dy in coarse_ring_candidates(R_STEP_COARSE, R_MAX, ANGLE_PRIORITY_DEG):
# # #         print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
# # #         if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0)):
# # #             return dx, dy, 0.0

# # #     # C) yaw-only (small) THEN very small translations
# # #     print("\n[SEARCH-C] rotation as last resort (small yaws first)")
# # #     for yaw_deg in YAW_SMALL_DEG:
# # #         yaw_rad = float(np.deg2rad(yaw_deg))

# # #         print(f"\n[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
# # #         if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad)):
# # #             return 0.0, 0.0, yaw_deg

# # #         # If yaw alone helps but not enough, allow tiny translations after yaw
# # #         for s in [0.05, 0.10, 0.20]:
# # #             for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
# # #                 print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
# # #                 if try_arm(arm, apply_base_se2(A_cur, float(dx), float(dy), yaw_rad)):
# # #                     return float(dx), float(dy), yaw_deg

# # #     # D) final fallback yaws (includes 180)
# # #     print("\n[SEARCH-D] final yaw fallback (90/-90/180)")
# # #     for yaw_deg in YAW_FALLBACK_DEG:
# # #         yaw_rad = float(np.deg2rad(yaw_deg))
# # #         print(f"[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
# # #         if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad)):
# # #             return 0.0, 0.0, yaw_deg

# # #     return None


# # # def main() -> int:
# # #     A_target = (Rz(TARGET_YAW_DEG) @ A).astype(float)
# # #     A_cur = A_target.copy()

# # #     cfg = ReachyConfig(host=HOST)
# # #     client = ReachyClient(cfg)
# # #     client.connect()
# # #     client.turn_on_all()

# # #     arm = client._get_arm(SIDE)
# # #     base = BaseController(client=client, world=None)

# # #     print("[BASE] (optional) Resetting odometry once at start...")
# # #     base.reset_odometry()
# # #     time.sleep(0.5)

# # #     try:
# # #         print("\n[1] First try without moving base")
# # #         if try_arm(arm, A_cur):
# # #             print("[SUCCESS] Reached without base move.")
# # #             return 0

# # #         print("\n[2] Unreachable -> translation-first base-assist search")
# # #         best = find_base_assist_translation_first(arm, A_cur)
# # #         if best is None:
# # #             print("[FAIL] No feasible base offset found.")
# # #             return 1

# # #         dx_goal, dy_goal, yaw_deg_goal = best
# # #         print(f"\n[BASE] Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

# # #         # EXECUTION ORDER (translation first, then rotation)
# # #         trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# # #         # update after translation
# # #         A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0)

# # #         yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# # #         # update after yaw
# # #         A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal))

# # #         print(f"[BASE] Executed trans_abs={trans_abs:.3f}m, yaw_abs={yaw_abs:.1f}deg")

# # #         print("\n[3] Retrying arm.goto after base assist")
# # #         if try_arm(arm, A_cur):
# # #             print("[SUCCESS] Reached after base assist.")
# # #             return 0

# # #         print("[FAIL] Feasible in search but failed after base motion (drift/slip/frames).")
# # #         return 1

# # #     finally:
# # #         try:
# # #             arm.turn_off_smoothly()
# # #         except Exception:
# # #             pass
# # #         try:
# # #             reachy = client.connect_reachy
# # #             reachy.mobile_base.turn_off()
# # #         except Exception:
# # #             pass
# # #         client.close()
# # #         print("Done.")


# # # if __name__ == "__main__":
# # #     raise SystemExit(main())



# # #!/usr/bin/env python3
# # from __future__ import annotations

# # import time
# # import numpy as np

# # from reachy2_stack.utils.utils_dataclass import ReachyConfig
# # from reachy2_stack.core.client import ReachyClient
# # from reachy2_stack.control.base import BaseController

# # # ============================== CONFIG =======================================
# # HOST = "192.168.1.71"
# # SIDE = "left"

# # # Rotate the TARGET itself around Z in the base frame (optional)
# # TARGET_YAW_DEG = 180.0

# # # Target pose (base <- ee)
# # A = np.array(
# #     [
# #         [0, 0, -1, 0.4],
# #         [0, 1,  0, -0.4],
# #         [1, 0,  0,  0.1],
# #         [0, 0,  0,  1.0],
# #     ],
# #     dtype=float,
# # )

# # # --- Safety / budgets ---
# # MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
# # MAX_BASE_TOTAL_YAW_DEG = 180.0    # max total yaw (deg)

# # MAX_STEP_TRANS = 0.30             # clamp per translate_by (m)
# # MAX_STEP_YAW_DEG = 30.0           # clamp per rotate_by (deg)

# # ARM_DURATION = 2.0

# # # --- Strategy 1 (your "not commented out"): translation-first, rotation last ---
# # LINE_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
# # AXIS_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
# # DIAG_FACTOR = 0.7

# # YAW_SMALL_DEG = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0]
# # YAW_FALLBACK_DEG = [90.0, -90.0, 180.0]

# # R_STEP_COARSE = 0.20
# # R_MAX = 2.0
# # ANGLE_PRIORITY_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]

# # # --- Strategy 2 (your "commented out"): yaw+translation SE(2) scan ---
# # # (Used ONLY if strategy-1 finds no candidate)
# # SE2_R_STEP = 0.05
# # SE2_R_MAX = 2.0
# # SE2_ANGLES = 16
# # SE2_YAW_LIST_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180]
# # # ============================================================================


# # def wrap180(deg: float) -> float:
# #     return (deg + 180.0) % 360.0 - 180.0


# # def Rz(deg: float) -> np.ndarray:
# #     th = np.deg2rad(float(deg))
# #     c, s = float(np.cos(th)), float(np.sin(th))
# #     R = np.eye(4, dtype=float)
# #     R[0, 0] = c
# #     R[0, 1] = -s
# #     R[1, 0] = s
# #     R[1, 1] = c
# #     return R


# # # ------------------------- Base motion models -------------------------
# # def T_translate_then_rotate(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# #     """
# #     Model for Strategy-1: base executes translate_by(dx,dy) THEN rotate_by(yaw)
# #     => T = Trans(dx,dy) @ Rot(yaw)
# #     """
# #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

# #     Tt = np.eye(4, dtype=float)
# #     Tt[0, 3] = float(dx)
# #     Tt[1, 3] = float(dy)

# #     R = np.eye(4, dtype=float)
# #     R[0, 0] = c
# #     R[0, 1] = -s
# #     R[1, 0] = s
# #     R[1, 1] = c

# #     return Tt @ R


# # def T_rotate_then_translate(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
# #     """
# #     Model for Strategy-2: base executes rotate_by(yaw) THEN translate_by(dx,dy)
# #     => T = Rot(yaw) @ Trans(dx,dy)
# #     (This matches the typical "rotate then move" execution order.)
# #     """
# #     c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

# #     R = np.eye(4, dtype=float)
# #     R[0, 0] = c
# #     R[0, 1] = -s
# #     R[1, 0] = s
# #     R[1, 1] = c

# #     Tt = np.eye(4, dtype=float)
# #     Tt[0, 3] = float(dx)
# #     Tt[1, 3] = float(dy)

# #     return R @ Tt


# # def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float, order: str) -> np.ndarray:
# #     """
# #     Update target pose expressed in NEW base frame:
# #         A_new = inv(T) @ A_old
# #     where T is the base motion in world/odom.
# #     order: "tr" (translate then rotate) or "rt" (rotate then translate)
# #     """
# #     if order == "tr":
# #         T = T_translate_then_rotate(dx, dy, yaw_rad)
# #     elif order == "rt":
# #         T = T_rotate_then_translate(dx, dy, yaw_rad)
# #     else:
# #         raise ValueError(f"Unknown order={order!r}")
# #     return np.linalg.inv(T) @ A_cur


# # # ------------------------- Arm + base helpers -------------------------
# # def try_arm(arm, A_try: np.ndarray) -> bool:
# #     resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
# #     gid = getattr(resp, "id", None)
# #     ok = gid is not None and gid != -1
# #     print("[ARM] goto id:", gid, "=>", "OK" if ok else "NO")
# #     return ok


# # def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
# #     remaining_dx = float(dx_goal)
# #     remaining_dy = float(dy_goal)
# #     total = 0.0

# #     while True:
# #         dist = float(np.hypot(remaining_dx, remaining_dy))
# #         if dist < 1e-6:
# #             break

# #         s = 1.0
# #         if dist > MAX_STEP_TRANS:
# #             s = MAX_STEP_TRANS / dist

# #         sx = remaining_dx * s
# #         sy = remaining_dy * s

# #         print(f"[BASE] translate_by(x={sx:+.3f}, y={sy:+.3f})")
# #         base.translate_by(x=float(sx), y=float(sy), wait=True)

# #         step_dist = float(np.hypot(sx, sy))
# #         total += step_dist
# #         if total > MAX_BASE_TOTAL_TRANS:
# #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

# #         remaining_dx -= sx
# #         remaining_dy -= sy

# #     return total


# # def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
# #     remaining = float(yaw_deg_goal)
# #     total_abs = 0.0

# #     while abs(remaining) > 1e-3:
# #         step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
# #         print(f"[BASE] rotate_by(theta={step:.1f} deg)")
# #         base.rotate_by(theta=step, wait=True, degrees=True)

# #         total_abs += abs(step)
# #         if total_abs > MAX_BASE_TOTAL_YAW_DEG:
# #             raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

# #         remaining -= step

# #     return total_abs


# # # ========================== STRATEGY 1 ======================================
# # def translation_priority_candidates(A_cur: np.ndarray) -> list[tuple[float, float]]:
# #     p = A_cur[:2, 3].astype(float)
# #     n = float(np.linalg.norm(p))
# #     if n < 1e-9:
# #         u = np.array([1.0, 0.0], dtype=float)
# #     else:
# #         u = p / n

# #     cands: list[tuple[float, float]] = []

# #     # 1) along target direction
# #     for s in LINE_STEP_TRIES:
# #         dx = float(u[0] * s)
# #         dy = float(u[1] * s)
# #         cands.append((dx, dy))

# #     # 1b) opposite direction
# #     for s in LINE_STEP_TRIES[:6]:
# #         dx = float(-u[0] * s)
# #         dy = float(-u[1] * s)
# #         cands.append((dx, dy))

# #     # 2) axis
# #     for s in AXIS_STEP_TRIES:
# #         for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
# #             cands.append((float(dx), float(dy)))

# #     # 3) diagonals
# #     for s in AXIS_STEP_TRIES:
# #         a = float(s)
# #         b = float(DIAG_FACTOR * s)
# #         for dx, dy in [(a, b), (a, -b), (-a, b), (-a, -b)]:
# #             cands.append((dx, dy))

# #     # filter + dedup
# #     seen = set()
# #     uniq: list[tuple[float, float]] = []
# #     for dx, dy in cands:
# #         if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
# #             continue
# #         key = (round(dx, 4), round(dy, 4))
# #         if key not in seen:
# #             seen.add(key)
# #             uniq.append((dx, dy))
# #     return uniq


# # def coarse_ring_candidates(r_step: float, r_max: float, angle_priority_deg: list[int]) -> list[tuple[float, float]]:
# #     radii = np.arange(r_step, r_max + 1e-9, r_step)
# #     cands: list[tuple[float, float]] = []
# #     for r in radii:
# #         for ang_deg in angle_priority_deg:
# #             th = np.deg2rad(float(ang_deg))
# #             dx = float(r * np.cos(th))
# #             dy = float(r * np.sin(th))
# #             if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
# #                 cands.append((dx, dy))
# #     return cands


# # def find_base_assist_strategy1(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
# #     """
# #     Strategy-1 (translation-first):
# #       A) translation-only (yaw=0) prioritized
# #       B) translation-only coarse rings
# #       C) small yaw last (optionally with tiny translations)
# #       D) final yaw fallback (90/-90/180)
# #     Returns (dx, dy, yaw_deg) or None.
# #     NOTE: Feasibility test uses apply_base_se2 with order="tr".
# #     """
# #     print("\n[STRAT-1 / A] translation-only (yaw=0) prioritized")
# #     for dx, dy in translation_priority_candidates(A_cur):
# #         print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
# #         if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
# #             return dx, dy, 0.0

# #     print("\n[STRAT-1 / B] translation-only coarse rings (yaw=0)")
# #     for dx, dy in coarse_ring_candidates(R_STEP_COARSE, R_MAX, ANGLE_PRIORITY_DEG):
# #         print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
# #         if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
# #             return dx, dy, 0.0

# #     print("\n[STRAT-1 / C] rotation as last resort (small yaws first)")
# #     for yaw_deg in YAW_SMALL_DEG:
# #         yaw_rad = float(np.deg2rad(yaw_deg))

# #         print(f"\n[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
# #         if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
# #             return 0.0, 0.0, yaw_deg

# #         for s in [0.05, 0.10, 0.20]:
# #             for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
# #                 print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
# #                 if try_arm(arm, apply_base_se2(A_cur, float(dx), float(dy), yaw_rad, order="tr")):
# #                     return float(dx), float(dy), yaw_deg

# #     print("\n[STRAT-1 / D] final yaw fallback (90/-90/180)")
# #     for yaw_deg in YAW_FALLBACK_DEG:
# #         yaw_rad = float(np.deg2rad(yaw_deg))
# #         print(f"[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
# #         if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
# #             return 0.0, 0.0, yaw_deg

# #     return None


# # # ========================== STRATEGY 2 ======================================
# # def find_base_assist_strategy2(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
# #     """
# #     Strategy-2 (fallback): SE(2) scan over yaw + ring translations (your older commented approach).
# #     Returns (dx, dy, yaw_deg) or None.
# #     NOTE: Feasibility test uses apply_base_se2 with order="rt" (rotate then translate).
# #     """
# #     print("\n[STRAT-2] SE(2) scan over (yaw, r, angle)")
# #     radii = np.arange(SE2_R_STEP, SE2_R_MAX + 1e-9, SE2_R_STEP)

# #     for yaw_deg in SE2_YAW_LIST_DEG:
# #         yaw_rad = float(np.deg2rad(yaw_deg))

# #         for r in radii:
# #             for i in range(SE2_ANGLES):
# #                 ang = 2.0 * np.pi * (i / SE2_ANGLES)
# #                 dx = float(r * np.cos(ang))
# #                 dy = float(r * np.sin(ang))

# #                 if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
# #                     continue

# #                 print(f"[TRY] yaw={yaw_deg:>4.0f} dx={dx:+.3f} dy={dy:+.3f}")
# #                 A_try = apply_base_se2(A_cur, dx, dy, yaw_rad, order="rt")
# #                 if try_arm(arm, A_try):
# #                     return dx, dy, float(yaw_deg)

# #     return None


# # # ============================== MAIN =========================================
# # def main() -> int:
# #     A_target = (Rz(TARGET_YAW_DEG) @ A).astype(float)
# #     A_cur = A_target.copy()

# #     cfg = ReachyConfig(host=HOST)
# #     client = ReachyClient(cfg)
# #     client.connect()
# #     client.turn_on_all()

# #     arm = client._get_arm(SIDE)
# #     base = BaseController(client=client, world=None)

# #     print("[BASE] (optional) Resetting odometry once at start...")
# #     base.reset_odometry()
# #     time.sleep(0.5)

# #     try:
# #         print("\n[0] Try arm without moving base")
# #         if try_arm(arm, A_cur):
# #             print("[SUCCESS] Reached without base move.")
# #             return 0

# #         # ---------- Strategy 1 ----------
# #         print("\n[1] Strategy-1: translation-first search")
# #         best = find_base_assist_strategy1(arm, A_cur)

# #         # If strategy-1 fails to find ANY candidate, fall back to strategy-2
# #         if best is None:
# #             print("\n[1->2] Strategy-1 found no solution. Falling back to Strategy-2 (yaw+translation scan).")
# #             best = find_base_assist_strategy2(arm, A_cur)
# #             if best is None:
# #                 print("[FAIL] No feasible base offset found in either strategy.")
# #                 return 1

# #             dx_goal, dy_goal, yaw_deg_goal = best
# #             print(f"\n[BASE] (Strategy-2) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

# #             # Execute Strategy-2 order: rotate THEN translate (matches order="rt")
# #             yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# #             A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="rt")

# #             trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# #             A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="rt")

# #             print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")

# #         else:
# #             dx_goal, dy_goal, yaw_deg_goal = best
# #             print(f"\n[BASE] (Strategy-1) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

# #             # Execute Strategy-1 order: translate THEN rotate (matches order="tr")
# #             trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
# #             A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="tr")

# #             yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
# #             A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="tr")

# #             print(f"[BASE] Executed trans_abs={trans_abs:.3f}m, yaw_abs={yaw_abs:.1f}deg")

# #         # ---------- Final retry ----------
# #         print("\n[2] Retrying arm.goto after base assist")
# #         if try_arm(arm, A_cur):
# #             print("[SUCCESS] Reached after base assist.")
# #             return 0

# #         print("[FAIL] Feasible in search but failed after base motion (drift/slip/frames).")
# #         return 1

# #     finally:
# #         try:
# #             arm.turn_off_smoothly()
# #         except Exception:
# #             pass
# #         try:
# #             reachy = client.connect_reachy
# #             reachy.mobile_base.turn_off()
# #         except Exception:
# #             pass
# #         client.close()
# #         print("Done.")


# # if __name__ == "__main__":
# #     raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")  # or your repo root


import time
import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.base import BaseController


# If you want to test world-frame too (optional):
# from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig


def main() -> int:
    HOST = "192.168.1.71"
    SIDE = "right"

    # A target pose in BASE frame (base <- ee)
    # Replace with your real pose(s)

    theta = np.deg2rad(0.0)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 0.0],
        [np.sin(theta),  np.cos(theta), 0.0, 0.0],
        [0.0,            0.0,           1.0, 0.0],
        [0.0,            0.0,           0.0, 1.0],
    ])
    A = np.array(
        [
            [0, 0, -1, -1.4],
            [0, 1,  0, -0.4],
            [1, 0,  0,  0.1],
            [0, 0,  0,  1.0],
        ],
        dtype=float,
    )
    T_base_ee = Rz @ A



    # --- Connect client ---
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()
    

    # --- Create controller (no world model needed for base-frame tests) ---
    arm = ArmController(client=client, side=SIDE, world=None)
    
    base = BaseController(client=client, world=None)
    # base.reset_odometry()

    try:

        print("\n==============================")
        print("start of the test")
        print("==============================")
        ok = arm.goto_pose_base_with_base_assist(
            T_base_ee=T_base_ee
        )
        print("Result:", "SUCCESS" if ok else "FAIL")
        time.sleep(1.0)


        return 0 if (ok) else 1

    finally:
        try:
            print("turning off.")
            client._get_arm(SIDE).turn_off_smoothly()
            
        except Exception:
            pass
        try:
            reachy = client.connect_reachy
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()
        print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())



