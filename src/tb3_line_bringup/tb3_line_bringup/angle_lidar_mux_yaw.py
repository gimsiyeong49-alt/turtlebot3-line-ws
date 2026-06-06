#!/usr/bin/env python3
"""
angle_lidar_mux  –  직각 우회 장애물 회피 mux
  - lateral_dist_m  : 22cm 옆으로 이동
  - align_back_deg  : 85° 역회전
  - SEARCH_FORWARD 중 측면 LiDAR 거리가 boundary_max_dist 초과하면 STOP
    (트랙 밖으로 벗어남 감지)
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def fmt(x):
    return f'{x:.2f}' if math.isfinite(x) else 'inf'


class AngleLidarMux(Node):

    def __init__(self):
        super().__init__('angle_lidar_mux')
        self._declare_params()

        self.line_cmd        = Twist()
        self.last_line_time  = 0.0
        self.front_min       = float('inf')
        self.boundary_min    = float('inf')   # ★ 측면 경계 최근접 거리
        self.side_obs_dist   = float('inf')   # ★ 장애물 측 근접 거리
        self.avoidance_done  = False              # ★ 회피 1회 완료 잠금
        self.path_obstacle   = False
        self.detect_count    = 0

        self.state           = 'FOLLOW'
        self.state_time      = time.time()
        self.side            = 1.0

        self.line_good_count = 0
        self.last_pub_v      = 0.0
        self.last_pub_w      = 0.0
        self.last_log_time   = 0.0

        self.traffic_stop_until = 0.0
        self.prev_traffic = False
        self.traffic_stop_count = 0

        # IMU/odom yaw feedback for accurate 90-degree turns
        self.imu_yaw = None
        self.odom_yaw = None
        self._yaw_prev = None
        self._yaw_unwrapped = 0.0
        self.turn_target_yaw = None
        self.turn_stable_count = 0
        self.turn_next_state = None
        self.turn_settle_until = 0.0

        self.pub = self.create_publisher(
            Twist, self.gstr('cmd_vel_topic'), 10)
        self.line_sub = self.create_subscription(
            Twist, self.gstr('line_cmd_topic'), self._line_cb, 10)

        self.traffic_sub = self.create_subscription(
            Bool, self.gstr('traffic_topic'), self._traffic_cb, 10)

        self.imu_sub = self.create_subscription(
            Imu, self.gstr('imu_topic'), self._imu_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.gstr('odom_topic'), self._odom_cb, 10)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.gstr('scan_topic'), self._scan_cb, scan_qos)

        self.timer = self.create_timer(self.g('timer_period'), self._timer_cb)
        self.get_logger().info('angle_lidar_mux (rectangular detour) started')
        self.get_logger().info(
            f"line={self.gstr('line_cmd_topic')}  "
            f"cmd={self.gstr('cmd_vel_topic')}  "
            f"scan={self.gstr('scan_topic')}"
        )

    def _declare_params(self):
        self.declare_parameter('line_cmd_topic', '/cmd_vel_line')
        self.declare_parameter('cmd_vel_topic',  '/cmd_vel')
        self.declare_parameter('scan_topic',     '/scan')
        self.declare_parameter('imu_topic',      '/imu')
        self.declare_parameter('odom_topic',     '/odom')

        # Yaw feedback turn control
        self.declare_parameter('use_imu_yaw',       True)
        self.declare_parameter('yaw_turn_enable',   True)
        self.declare_parameter('yaw_turn_kp',       2.4)
        self.declare_parameter('yaw_turn_tol_deg',  1.5)
        self.declare_parameter('yaw_turn_stable_n', 4)
        self.declare_parameter('yaw_turn_w_min',    0.16)
        self.declare_parameter('yaw_turn_timeout_margin', 1.2)
        self.declare_parameter('turn_settle_time', 0.20)

        # Traffic light Bool input only. Image processing is external.
        self.declare_parameter('traffic_bool_enable', True)
        self.declare_parameter('traffic_topic', '/traffic_light')
        self.declare_parameter('traffic_stop_time', 4.0)
        self.declare_parameter('traffic_stop_max_count', 2)

        # 경로 투영 검출
        self.declare_parameter('path_max_dist',   0.70)
        self.declare_parameter('path_steps',      12)
        self.declare_parameter('path_width_m',    0.22)
        self.declare_parameter('path_min_x',      0.08)

        # forward-cone fallback
        self.declare_parameter('front_deg',       22.0)
        self.declare_parameter('enter_dist',      0.40)
        self.declare_parameter('early_x',         0.40)
        self.declare_parameter('early_y_abs',     0.22)

        # 공통
        self.declare_parameter('hard_stop_dist',  0.13)
        self.declare_parameter('detect_frames',   2)
        self.declare_parameter('min_valid_range', 0.05)
        self.declare_parameter('max_valid_range', 2.20)
        self.declare_parameter('force_side',      1.0)

        # 회전 각도
        self.declare_parameter('turn_out_deg',    90.0)
        self.declare_parameter('align_back_deg',  85.0)   # ★ 85°
        self.declare_parameter('turn_w',          0.70)
        self.declare_parameter('turn_v',          0.000)

        # LATERAL_MOVE
        self.declare_parameter('lateral_dist_m',  0.25)   # ★ 22cm
        self.declare_parameter('lateral_v',       0.12)

        # FORWARD_PASS
        self.declare_parameter('forward_pass_time', 3.0)
        self.declare_parameter('forward_pass_v',    0.15)
        self.declare_parameter('return_arc_time',   1.20)  # 복귀 호 시간(초)
        self.declare_parameter('return_arc_v',      0.12)  # 복귀 호 전진 속도
        self.declare_parameter('return_arc_w',      0.25)  # 복귀 호 각속도
        self.declare_parameter('pass_steer_thresh', 0.50)
        self.declare_parameter('pass_steer_w',      0.15)

        # SEARCH_FORWARD
        self.declare_parameter('search_time',        15.0)
        self.declare_parameter('search_v',           0.12)

        # ★ 경계 이탈 감지 (SEARCH_FORWARD 중)
        # 양쪽 측면(±y) 중 가장 가까운 벽까지 거리가
        # boundary_max_dist 초과하면 트랙 밖으로 판단 → STOP
        self.declare_parameter('boundary_max_dist',  1.00)  # (m)
        self.declare_parameter('boundary_x_min',    -0.20)  # 측면 검사 x 범위
        self.declare_parameter('boundary_x_max',     0.20)

        # 라인 복귀 판정
        self.declare_parameter('cmd_timeout',        0.35)
        self.declare_parameter('return_min_v',       0.080)
        self.declare_parameter('return_max_abs_w',   0.250)
        self.declare_parameter('line_good_frames',   4)

        # 슬루
        self.declare_parameter('timer_period',    0.08)
        self.declare_parameter('linear_slew',     0.60)
        self.declare_parameter('angular_slew',    2.80)
        self.declare_parameter('log_interval',    0.20)

    def g(self, n):    return float(self.get_parameter(n).value)
    def gi(self, n):   return int(self.get_parameter(n).value)
    def gb(self, n):   return bool(self.get_parameter(n).value)
    def gstr(self, n): return str(self.get_parameter(n).value)

    def _quat_to_yaw(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _imu_cb(self, msg: Imu):
        if msg.orientation_covariance[0] < 0.0:
            self.imu_yaw = None
            return
        self.imu_yaw = self._quat_to_yaw(msg.orientation)

    def _odom_cb(self, msg: Odometry):
        self.odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

    def _get_yaw(self):
        if self.gb('use_imu_yaw') and self.imu_yaw is not None:
            return self.imu_yaw
        if self.odom_yaw is not None:
            return self.odom_yaw
        return None

    def _get_yaw_unwrapped(self):
        y = self._get_yaw()
        if y is None:
            return None
        if self._yaw_prev is None:
            self._yaw_prev = y
            self._yaw_unwrapped = y
            return self._yaw_unwrapped
        dy = wrap_pi(y - self._yaw_prev)
        self._yaw_unwrapped += dy
        self._yaw_prev = y
        return self._yaw_unwrapped

    def _start_yaw_turn(self, delta_rad: float):
        cur = self._get_yaw_unwrapped()
        if cur is None:
            self.turn_target_yaw = None
            self.turn_stable_count = 0
            self.get_logger().warn('YAW_TURN unavailable -> fallback to timed turn')
            return
        self.turn_target_yaw = cur + float(delta_rad)
        self.turn_stable_count = 0
        self.last_pub_w = 0.0
        self.get_logger().warn(
            f'YAW_TURN target delta={math.degrees(delta_rad):.1f}deg '
            f'cur={math.degrees(cur):.1f}deg target={math.degrees(self.turn_target_yaw):.1f}deg'
        )

    def _yaw_turn_step(self, now, next_state: str, max_time: float):
        if (not self.gb('yaw_turn_enable')) or self.turn_target_yaw is None:
            return False

        cur = self._get_yaw_unwrapped()
        if cur is None:
            return False

        err = self.turn_target_yaw - cur
        tol = math.radians(self.g('yaw_turn_tol_deg'))

        if abs(err) <= tol:
            self.turn_stable_count += 1
            # 목표각 도달 시 slew를 거치면 잔류 angular.z 때문에 다음 상태에서 더 돌아감
            self._publish_hard_stop()
            self._log(now, 0.0, 0.0)
            if self.turn_stable_count >= self.gi('yaw_turn_stable_n'):
                self.turn_target_yaw = None
                self.turn_stable_count = 0
                self.turn_next_state = next_state
                self.turn_settle_until = now + self.g('turn_settle_time')
                self._set_state('TURN_SETTLE', now)
            return True

        self.turn_stable_count = 0
        # Safety timeout: do not spin forever if yaw feedback freezes.
        if (now - self.state_time) > max_time + self.g('yaw_turn_timeout_margin'):
            self.get_logger().warn(
                f'YAW_TURN timeout err={math.degrees(err):.1f}deg -> {next_state}'
            )
            self.turn_target_yaw = None
            self._set_state(next_state, now)
            return True

        w = self.g('yaw_turn_kp') * err
        w = clamp(w, -abs(self.g('turn_w')), abs(self.g('turn_w')))
        w_min = abs(self.g('yaw_turn_w_min'))
        if abs(w) < w_min:
            w = w_min * (1.0 if w >= 0.0 else -1.0)

        v = self.g('turn_v')
        self._publish(v, w)
        self._log(now, v, w)
        return True

    def _traffic_cb(self, msg: Bool):
        if self.g('traffic_bool_enable') <= 0.5:
            self.prev_traffic = bool(msg.data)
            return

        now = time.time()
        red = bool(msg.data)
        max_count = self.gi('traffic_stop_max_count')

        # false -> true 순간에만 신호등 정지 카운트
        if red and not self.prev_traffic:
            if self.traffic_stop_count < max_count:
                self.traffic_stop_count += 1
                self.traffic_stop_until = now + self.g('traffic_stop_time')
                self.get_logger().warn(
                    f'TRAFFIC BOOL STOP #{self.traffic_stop_count}/{max_count} '
                    f'{self.g("traffic_stop_time"):.1f}s'
                )
            else:
                self.get_logger().warn(
                    f'TRAFFIC ignored count={self.traffic_stop_count}/{max_count}'
                )

        self.prev_traffic = red

    def _traffic_stop_active(self, now):
        if self.g('traffic_bool_enable') <= 0.5:
            return False
        return now < self.traffic_stop_until

    def _publish_hard_stop(self):
        self.last_pub_v = 0.0
        self.last_pub_w = 0.0
        self.pub.publish(Twist())

    def _line_cb(self, msg: Twist):
        self.line_cmd       = msg
        self.last_line_time = time.time()

    def _scan_cb(self, msg: LaserScan):
        r_min = max(float(msg.range_min), self.g('min_valid_range'))
        r_max = min(float(msg.range_max), self.g('max_valid_range'))

        bx_min = self.g('boundary_x_min')
        bx_max = self.g('boundary_x_max')

        points       = []
        front_min    = float('inf')
        boundary_min = float('inf')
        front_rad    = math.radians(self.g('front_deg'))

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= r_min or r >= r_max:
                continue
            a = msg.angle_min + i * msg.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append((x, y))

            if abs(a) <= front_rad:
                front_min = min(front_min, r)

            # ★ 측면 경계: 로봇 옆(±y)의 벽까지 거리
            # x 범위를 좁게 잡아 정면/후면 포인트 제외
            if bx_min <= x <= bx_max:
                boundary_min = min(boundary_min, abs(y))

        # 장애물 측(self.side 방향) 근접 거리 - 트럭 끝났는지 판단용
        side_obs = float('inf')
        for (x, y) in points:
            if 0.0 <= x <= 0.60:  # 로봇 앞 0~60cm 범위
                proj = y * self.side  # side>0이면 y, side<0이면 -y
                if proj > 0.0:
                    side_obs = min(side_obs, proj)
        self.front_min    = front_min
        self.boundary_min = boundary_min
        self.side_obs_dist = side_obs

        detected = (
            self._path_projected_detection(points)
            if self.g('path_max_dist') > 0.0
            else self._forward_cone_detection(points)
        )
        self.detect_count  = (self.detect_count + 1) if detected else 0
        self.path_obstacle = self.detect_count >= self.gi('detect_frames')

    def _path_projected_detection(self, points) -> bool:
        now     = time.time()
        total_d = self.g('path_max_dist')
        steps   = max(self.gi('path_steps'), 2)
        half_w  = self.g('path_width_m')
        min_x   = self.g('path_min_x')
        ds      = total_d / steps

        if now - self.last_line_time > self.g('cmd_timeout'):
            k = 0.0
        else:
            v = float(self.line_cmd.linear.x)
            w = float(self.line_cmd.angular.z)
            k = (w / v) if abs(v) > 0.01 else 0.0

        px, py, pth = 0.0, 0.0, 0.0
        segs = []
        for _ in range(steps):
            nx  = px + ds * math.cos(pth)
            ny  = py + ds * math.sin(pth)
            segs.append((px, py, nx, ny))
            px, py, pth = nx, ny, pth + k * ds

        for (ox, oy) in points:
            if ox < min_x:
                continue
            for (ax, ay, bx, by) in segs:
                dx, dy = bx - ax, by - ay
                seg2   = dx * dx + dy * dy
                if seg2 < 1e-9:
                    continue
                t = ((ox-ax)*dx + (oy-ay)*dy) / seg2
                if not 0.0 <= t <= 1.0:
                    continue
                if math.hypot(ox-(ax+t*dx), oy-(ay+t*dy)) < half_w:
                    return True
        return False

    def _forward_cone_detection(self, points) -> bool:
        enter = self.g('enter_dist')
        ex    = self.g('early_x')
        ey    = self.g('early_y_abs')
        fr    = math.radians(self.g('front_deg'))
        for (x, y) in points:
            if abs(math.atan2(y, x)) <= fr and math.hypot(x, y) < enter:
                return True
            if 0.0 <= x <= ex and abs(y) <= ey:
                return True
        return False

    # ★ SEARCH_FORWARD 중 트랙 이탈 판정
    def _out_of_boundary(self) -> bool:
        return self.boundary_min > self.g('boundary_max_dist')

    def _turn_out_time(self):
        return math.radians(abs(self.g('turn_out_deg'))) / max(0.05, abs(self.g('turn_w')))

    def _align_back_time(self):
        return math.radians(abs(self.g('align_back_deg'))) / max(0.05, abs(self.g('turn_w')))

    def _lateral_time(self):
        return self.g('lateral_dist_m') / max(0.01, self.g('lateral_v'))

    def _line_is_stable(self, now):
        if now - self.last_line_time > self.g('cmd_timeout'):
            return False
        return (float(self.line_cmd.linear.x) >= self.g('return_min_v') and
                abs(float(self.line_cmd.angular.z)) <= self.g('return_max_abs_w'))

    def _line_return_ready(self, now):
        self.line_good_count = (self.line_good_count+1) if self._line_is_stable(now) else 0
        return self.line_good_count >= self.gi('line_good_frames')

    def _set_state(self, name, now):
        self.state = name; self.state_time = now
        self.line_good_count = 0; self.detect_count = 0
        self.turn_target_yaw = None
        self.turn_stable_count = 0

        if name == 'TURN_OUT':
            self._start_yaw_turn(self.side * math.radians(abs(self.g('turn_out_deg'))))
        elif name == 'ALIGN_BACK':
            self._start_yaw_turn(-self.side * math.radians(abs(self.g('align_back_deg'))))

        self.get_logger().warn(
            f'STATE → {name}  front={fmt(self.front_min)}  '
            f'boundary={fmt(self.boundary_min)}  '
            f'side={"L" if self.side > 0 else "R"}')

    def _apply_slew(self, v, w):
        dt = self.g('timer_period')
        v2 = self.last_pub_v + clamp(v-self.last_pub_v, -self.g('linear_slew')*dt,  self.g('linear_slew')*dt)
        w2 = self.last_pub_w + clamp(w-self.last_pub_w, -self.g('angular_slew')*dt, self.g('angular_slew')*dt)
        self.last_pub_v, self.last_pub_w = v2, w2
        return v2, w2

    def _publish(self, v, w):
        v, w = self._apply_slew(v, w)
        msg = Twist(); msg.linear.x = float(v); msg.angular.z = float(w)
        self.pub.publish(msg)

    def _publish_line(self, now):
        if now - self.last_line_time > self.g('cmd_timeout'):
            self._publish(0.0, 0.0)
        else:
            self._publish(max(0.0, float(self.line_cmd.linear.x)),
                          float(self.line_cmd.angular.z))

    def _log(self, now, v, w):
        if now - self.last_log_time < self.g('log_interval'): return
        self.last_log_time = now
        self.get_logger().warn(
            f'{self.state}  front={fmt(self.front_min)}  '
            f'boundary={fmt(self.boundary_min)}  '
            f'path_obs={self.path_obstacle}  '
            f'v={v:.3f}  w={w:.3f}  line_ok={self.line_good_count}')

    def _timer_cb(self):
        now = time.time()
        elapsed = now - self.state_time

        # TRAFFIC BOOL STOP OVERRIDE
        # 빨간불이면 FOLLOW/회피 상태 상관없이 최우선 정지
        if self._traffic_stop_active(now):
            self.state_time = now
            self._publish_hard_stop()
            if now - self.last_log_time >= self.g('log_interval'):
                self.last_log_time = now
                left = max(0.0, self.traffic_stop_until - now)
                self.get_logger().warn(f'TRAFFIC_STOP active left={left:.1f}s state={self.state}')
            return

        # ⓪ TURN_SETTLE: 회전 끝난 뒤 잔류 각속도 제거용 짧은 정지
        if self.state == 'TURN_SETTLE':
            self._publish_hard_stop()
            if now >= self.turn_settle_until:
                nxt = self.turn_next_state or 'FOLLOW'
                self.turn_next_state = None
                self._set_state(nxt, now)
            return

        # ① FOLLOW
        if self.state == 'FOLLOW':
            if self.path_obstacle and not self.avoidance_done:
                self.side = 1.0 if self.g('force_side') >= 0.0 else -1.0
                self._set_state('TURN_OUT', now); return
            self._publish_line(now)
            self._log(now, float(self.line_cmd.linear.x), float(self.line_cmd.angular.z))
            return

        # ② TURN_OUT
        if self.state == 'TURN_OUT':
            if self._yaw_turn_step(now, 'LATERAL_MOVE', self._turn_out_time()):
                return
            if elapsed >= self._turn_out_time():
                self._set_state('LATERAL_MOVE', now); return
            v, w = self.g('turn_v'), self.side * self.g('turn_w')
            self._publish(v, w); self._log(now, v, w); return

        # ③ LATERAL_MOVE
        if self.state == 'LATERAL_MOVE':
            if self.front_min < self.g('hard_stop_dist'):
                self._publish(0.0, 0.0); return
            if elapsed >= self._lateral_time():
                self._set_state('ALIGN_BACK', now); return
            self._publish(self.g('lateral_v'), 0.0)
            self._log(now, self.g('lateral_v'), 0.0); return

        # ④ ALIGN_BACK
        if self.state == 'ALIGN_BACK':
            if self._yaw_turn_step(now, 'FORWARD_PASS', self._align_back_time()):
                return
            if elapsed >= self._align_back_time():
                self._set_state('FORWARD_PASS', now); return
            v, w = self.g('turn_v'), -self.side * self.g('turn_w')
            self._publish(v, w); self._log(now, v, w); return

        # ⑤ FORWARD_PASS
        if self.state == 'FORWARD_PASS':
            if self.front_min < self.g('hard_stop_dist'):
                self._publish(0.0, 0.0); self._log(now, 0.0, 0.0); return
            if elapsed >= self.g('forward_pass_time'):
                self._set_state('RETURN_ARC', now); return
            v = self.g('forward_pass_v')
            self._publish(v, 0.0); self._log(now, v, 0.0); return

        # ⑤-b RETURN_ARC : 트럭 끝난 후 라인 방향으로 호 그리며 복귀
        if self.state == 'RETURN_ARC':
            if self._line_return_ready(now):
                self.avoidance_done = True
                self._set_state('FOLLOW', now); self._publish_line(now); return
            if elapsed >= self.g('return_arc_time'):
                self._set_state('SEARCH_FORWARD', now); return
            # 트럭 아직 옆에 있으면 직진, 끝나면 호
            truck_gone = self.side_obs_dist > self.g('pass_steer_thresh')
            v = self.g('return_arc_v')
            w = -self.side * self.g('return_arc_w') if truck_gone else 0.0
            self._publish(v, w); self._log(now, v, w); return

        # ⑥ SEARCH_FORWARD ★ 경계 이탈 감지 추가
        if self.state == 'SEARCH_FORWARD':
            if self._line_return_ready(now):
                self.avoidance_done = True
                self._set_state('FOLLOW', now); self._publish_line(now); return
            if elapsed >= self.g('search_time'):
                self.get_logger().warn('SEARCH timeout → STOP')
                self._set_state('STOP', now); return
            if self._out_of_boundary():
                self.get_logger().warn(
                    f'SEARCH boundary exceeded '
                    f'(boundary={fmt(self.boundary_min)} > '
                    f'{self.g("boundary_max_dist"):.2f}) → STOP')
                self._set_state('STOP', now); return
            self._publish(self.g('search_v'), 0.0)
            self._log(now, self.g('search_v'), 0.0); return

        # ⑦ STOP
        if self.state == 'STOP':
            if self._line_return_ready(now):
                self._set_state('FOLLOW', now); self._publish_line(now); return
            self._publish(0.0, 0.0); self._log(now, 0.0, 0.0); return

        self._set_state('FOLLOW', now); self._publish_line(now)


def main(args=None):
    rclpy.init(args=args)
    node = AngleLidarMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
