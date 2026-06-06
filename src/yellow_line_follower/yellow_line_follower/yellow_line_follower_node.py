#!/usr/bin/env python3
# PATCHED v4: near_margin + first_image + feedforward(kff) 추가
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge


class YellowLineFollower(Node):
    def __init__(self):
        super().__init__('yellow_line_follower_node')

        self.declare_parameter('image_topic',   '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('avoid_enable', False)
        self.declare_parameter('vision_avoid_enable', True)
        self.declare_parameter('vision_obstacle_y_start_ratio', 0.46)
        self.declare_parameter('vision_obstacle_y_end_ratio',   0.84)
        self.declare_parameter('vision_obstacle_center_width_ratio', 0.48)
        self.declare_parameter('vision_obstacle_dark_v_max', 95)
        self.declare_parameter('vision_obstacle_sat_min', 55)
        self.declare_parameter('vision_obstacle_v_min', 40)
        self.declare_parameter('vision_obstacle_min_area', 1100)
        self.declare_parameter('vision_avoid_turn_speed', 0.42)
        self.declare_parameter('vision_avoid_forward_speed', 0.060)
        self.declare_parameter('vision_avoid_bypass_left_w', 0.18)
        self.declare_parameter('vision_avoid_recover_w', 0.20)
        self.declare_parameter('vision_avoid_turn_time', 0.22)
        self.declare_parameter('vision_avoid_bypass_min_time', 0.35)
        self.declare_parameter('vision_avoid_bypass_max_time', 0.85)
        self.declare_parameter('vision_avoid_recover_time', 0.35)
        self.declare_parameter('vision_avoid_log_interval', 0.20)

        self.declare_parameter('h_min', 10)
        self.declare_parameter('s_min', 35)
        self.declare_parameter('v_min', 50)
        self.declare_parameter('h_max', 55)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)

        self.declare_parameter('far_y_start_ratio',  0.60)
        self.declare_parameter('far_y_end_ratio',    0.70)
        self.declare_parameter('mid_y_start_ratio',  0.70)
        self.declare_parameter('mid_y_end_ratio',    0.82)
        self.declare_parameter('near_y_start_ratio', 0.82)
        self.declare_parameter('near_y_end_ratio',   0.98)

        self.declare_parameter('near_dist_m',          0.22)
        self.declare_parameter('mid_dist_m',           0.34)
        self.declare_parameter('far_dist_m',           0.58)
        self.declare_parameter('lookahead_straight_m', 0.32)
        self.declare_parameter('lookahead_turn_m',     0.12)

        self.declare_parameter('far_ignore_curve_px', 45.0)
        self.declare_parameter('far_max_jump_px',     85.0)
        self.declare_parameter('max_target_shift_px', 55.0)

        self.declare_parameter('fitline_y_start_ratio',      0.58)
        self.declare_parameter('fitline_y_end_ratio',        0.98)
        self.declare_parameter('fitline_target_y_ratio',     0.72)
        self.declare_parameter('fitline_min_area',           30)
        self.declare_parameter('fitline_blend',              0.65)
        self.declare_parameter('fitline_blend_low',          0.30)
        self.declare_parameter('fitline_force_curve_px',     45.0)
        self.declare_parameter('fitline_min_curvature_norm', 0.35)

        self.declare_parameter('curvature_decay', 0.75)

        self.declare_parameter('min_area_near', 25)
        self.declare_parameter('min_area_mid',  20)
        self.declare_parameter('min_area_far',  15)

        self.declare_parameter('base_speed', 0.135)
        self.declare_parameter('min_speed',  0.050)
        self.declare_parameter('max_speed',  0.200)

        self.declare_parameter('kp_target',         0.0020)
        self.declare_parameter('kd',                0.0009)
        self.declare_parameter('kff',               0.00015)  # [NEW] feedforward gain
        self.declare_parameter('max_angular',       1.20)
        self.declare_parameter('angular_slew_rate', 3.5)
        self.declare_parameter('center_offset_px',  0.0)

        # [NEW] 곡선 미래예측 보정: near-mid-far 흐름으로 target_x를 선제 보정
        self.declare_parameter('predictive_curve_enable', True)
        self.declare_parameter('predictive_curve_gain', 0.38)
        self.declare_parameter('predictive_curve_max_px', 24.0)
        self.declare_parameter('predictive_curve_min_px', 8.0)
        self.declare_parameter('predictive_curve_curv_min', 0.22)
        self.declare_parameter('predictive_curve_straight_px', 16.0)

        # [NEW] S자 이후 직선 진입 시 target_x 순간 튐 완화
        self.declare_parameter('target_smooth_alpha', 0.28)
        self.declare_parameter('target_smooth_max_step_px', 7.0)
        self.declare_parameter('straight_deadband_px', 10.0)
        self.declare_parameter('straight_curvature_thresh', 0.18)
        self.declare_parameter('straight_angular_scale', 0.45)

        # [CORNER_EXIT_RECENTER]
        # 코너 탈출 후 직진 진입에서 라인이 한쪽으로 밀릴 때
        # smoothing을 빠르게 풀고 최소 조향을 강제로 줘서 중앙선 복귀
        self.declare_parameter('corner_exit_recenter_enable', True)
        self.declare_parameter('recenter_curv_max', 0.22)
        self.declare_parameter('recenter_err_px', 35.0)
        self.declare_parameter('recenter_near_edge_px', 65.0)
        self.declare_parameter('recenter_smooth_alpha', 0.85)
        self.declare_parameter('recenter_smooth_max_step_px', 22.0)
        self.declare_parameter('recenter_min_w', 0.28)
        self.declare_parameter('recenter_max_w', 0.48)
        self.declare_parameter('recenter_speed', 0.115)

        self.declare_parameter('curve_slowdown',          0.55)
        self.declare_parameter('edge_slowdown',           0.25)
        self.declare_parameter('corner_speed_min',        0.095)
        self.declare_parameter('corner_speed_start_norm', 0.16)

        self.declare_parameter('lost_timeout',   0.35)
        self.declare_parameter('coast_time',     0.30)  # [NEW] 흰마커 통과 시간
        self.declare_parameter('coast_factor',   0.85)  # [NEW] coast 시 속도 비율
        self.declare_parameter('search_speed',   0.000)
        self.declare_parameter('search_angular', 0.30)

        self.declare_parameter('publish_debug',  False)

        # [LINE IGNORE] 노란 신호등/표지판이 노란 라인으로 잡히는 것 방지
        self.declare_parameter('line_ignore_yellow_signal_enable', True)
        self.declare_parameter('line_ignore_top', 0.38)
        self.declare_parameter('line_ignore_bottom', 0.70)
        self.declare_parameter('line_ignore_left', 0.00)
        self.declare_parameter('line_ignore_right', 0.46)

        # [SLOW SIGN] 왼쪽 ROI의 빨간 삼각 표지판 + 흰 내부 + 검정 글자 흔적 감지
        self.declare_parameter('slow_sign_enable', True)
        self.declare_parameter('slow_sign_topic', '/slow_sign')
        self.declare_parameter('slow_sign_process_hz', 3.0)
        self.declare_parameter('slow_sign_hold_time', 3.0)
        self.declare_parameter('slow_sign_max_speed', 0.105)
        self.declare_parameter('slow_sign_log_interval', 0.40)

        self.declare_parameter('slow_roi_top', 0.32)
        self.declare_parameter('slow_roi_bottom', 0.64)
        self.declare_parameter('slow_roi_left', 0.03)
        self.declare_parameter('slow_roi_right', 0.32)

        self.declare_parameter('slow_red_min', 75)
        self.declare_parameter('slow_red_margin_g', 12)
        self.declare_parameter('slow_red_margin_b', 12)
        self.declare_parameter('slow_min_pixels', 80)
        self.declare_parameter('slow_min_ratio_x1000', 3)
        self.declare_parameter('slow_min_contour_area', 50)
        self.declare_parameter('slow_min_box_w', 14)
        self.declare_parameter('slow_min_box_h', 14)
        self.declare_parameter('slow_reject_edge_px', 4)

        self.declare_parameter('slow_inner_check_enable', True)
        self.declare_parameter('slow_inner_margin_ratio', 0.18)
        self.declare_parameter('slow_white_min', 110)
        self.declare_parameter('slow_white_delta_max', 100)
        self.declare_parameter('slow_min_white_ratio_x1000', 80)
        self.declare_parameter('slow_black_max', 100)
        self.declare_parameter('slow_min_black_pixels', 5)
        self.declare_parameter('slow_min_black_ratio_x1000', 2)
        self.declare_parameter('slow_confirm_frames', 1)
        self.declare_parameter('slow_clear_frames', 3)

        # [TRAFFIC] 추가 카메라 구독 없이 현재 frame에서 신호등 ROI 감지
        self.declare_parameter('traffic_light_enable', True)
        self.declare_parameter('traffic_topic', '/traffic_light')
        self.declare_parameter('traffic_process_hz', 2.0)
        self.declare_parameter('traffic_log_interval', 0.50)
        self.declare_parameter('traffic_roi_top', 0.42)
        self.declare_parameter('traffic_roi_bottom', 0.62)
        self.declare_parameter('traffic_roi_left', 0.04)
        self.declare_parameter('traffic_roi_right', 0.43)
        self.declare_parameter('traffic_red_min', 115)
        self.declare_parameter('traffic_red_margin_g', 30)
        self.declare_parameter('traffic_red_margin_b', 30)
        self.declare_parameter('traffic_min_pixels', 350)
        self.declare_parameter('traffic_min_ratio_x1000', 50)
        self.declare_parameter('traffic_confirm_frames', 1)
        self.declare_parameter('traffic_clear_frames', 3)
        self.declare_parameter('log_interval',   0.25)
        self.declare_parameter('invert_angular', False)

        image_topic   = self.get_parameter('image_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.bridge = CvBridge()

        self.last_error          = 0.0
        self.last_time           = time.time()
        self.last_seen_time      = time.time()
        self.last_log_time       = 0.0
        self.last_angular        = 0.0
        self.last_turn_sign      = 1.0
        self.first_detection     = True
        self.last_curvature_norm = 0.0
        self.last_cx_near        = None
        self.last_target_x       = None  # [NEW] feedforward용
        self.filtered_target_x   = None  # [NEW] target_x smoothing용
        self.last_linear         = 0.0   # [NEW] coast용
        self.first_image         = True

        self.slow_sign_state = False
        self.slow_sign_until = 0.0
        self.slow_red_count = 0
        self.slow_clear_count = 0
        self.last_slow_process_time = 0.0
        self.last_slow_log_time = 0.0

        self.traffic_state = False
        self.traffic_red_count = 0
        self.traffic_clear_count = 0
        self.last_traffic_process_time = 0.0
        self.last_traffic_log_time = 0.0

        self.vision_avoid_state         = 'FOLLOW'
        self.vision_avoid_state_time    = time.time()
        self.vision_obstacle_seen       = False
        self.vision_obstacle_area       = 0.0
        self.vision_obstacle_cx         = None
        self.last_vision_avoid_log_time = 0.0

        self.cmd_pub   = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, '/line_follower/debug_image', 10)
        self.mask_pub  = self.create_publisher(Image, '/line_follower/yellow_mask',  10)
        self.slow_sign_pub = self.create_publisher(
            Bool, self.get_parameter('slow_sign_topic').value, 10
        )
        self.traffic_pub = self.create_publisher(
            Bool, self.get_parameter('traffic_topic').value, 10
        )

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.get_logger().info('yellow_line_follower_node started (patched v4 + kff)')
        self.get_logger().info(f'image_topic: {image_topic}')
        self.get_logger().info(f'cmd_vel_topic: {cmd_vel_topic}')

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def get_float(self, n): return float(self.get_parameter(n).value)
    def get_int(self,   n): return int(self.get_parameter(n).value)
    def get_bool(self,  n): return bool(self.get_parameter(n).value)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())


    def update_traffic_light_roi(self, frame, now):
        if not self.get_bool('traffic_light_enable'):
            return

        hz = self.get_float('traffic_process_hz')
        if hz > 0.0 and now - self.last_traffic_process_time < 1.0 / hz:
            return
        self.last_traffic_process_time = now

        h, w = frame.shape[:2]

        y1 = int(h * self.get_float('traffic_roi_top'))
        y2 = int(h * self.get_float('traffic_roi_bottom'))
        x1 = int(w * self.get_float('traffic_roi_left'))
        x2 = int(w * self.get_float('traffic_roi_right'))

        y1 = int(self.clamp(y1, 0, h - 1))
        y2 = int(self.clamp(y2, y1 + 1, h))
        x1 = int(self.clamp(x1, 0, w - 1))
        x2 = int(self.clamp(x2, x1 + 1, w))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        b = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        r = roi[:, :, 2].astype(np.int16)

        red_mask = (
            (r >= self.get_int('traffic_red_min')) &
            ((r - g) >= self.get_int('traffic_red_margin_g')) &
            ((r - b) >= self.get_int('traffic_red_margin_b'))
        )

        red_pixels = int(np.count_nonzero(red_mask))
        total_pixels = max(1, int(red_mask.size))
        ratio_x1000 = int(red_pixels * 1000 / total_pixels)

        raw_detected = (
            red_pixels >= self.get_int('traffic_min_pixels') and
            ratio_x1000 >= self.get_int('traffic_min_ratio_x1000')
        )

        if raw_detected:
            self.traffic_red_count += 1
            self.traffic_clear_count = 0
        else:
            self.traffic_clear_count += 1
            self.traffic_red_count = 0

        if self.traffic_red_count >= self.get_int('traffic_confirm_frames'):
            self.traffic_state = True

        if self.traffic_clear_count >= self.get_int('traffic_clear_frames'):
            self.traffic_state = False

        out = Bool()
        out.data = bool(self.traffic_state)
        self.traffic_pub.publish(out)

        if now - self.last_traffic_log_time >= self.get_float('traffic_log_interval'):
            self.last_traffic_log_time = now
            self.get_logger().warn(
                f'TRAFFIC_ROI state={self.traffic_state} raw={raw_detected} '
                f'pixels={red_pixels} ratio={ratio_x1000}/1000'
            )



    def update_slow_sign_roi(self, frame, now):
        if not self.get_bool('slow_sign_enable'):
            return

        hz = self.get_float('slow_sign_process_hz')
        if hz > 0.0 and now - self.last_slow_process_time < 1.0 / hz:
            return
        self.last_slow_process_time = now

        h, w = frame.shape[:2]

        y1 = int(h * self.get_float('slow_roi_top'))
        y2 = int(h * self.get_float('slow_roi_bottom'))
        x1 = int(w * self.get_float('slow_roi_left'))
        x2 = int(w * self.get_float('slow_roi_right'))

        y1 = int(self.clamp(y1, 0, h - 1))
        y2 = int(self.clamp(y2, y1 + 1, h))
        x1 = int(self.clamp(x1, 0, w - 1))
        x2 = int(self.clamp(x2, x1 + 1, w))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        b = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        r = roi[:, :, 2].astype(np.int16)

        red_mask = (
            (r >= self.get_int('slow_red_min')) &
            ((r - g) >= self.get_int('slow_red_margin_g')) &
            ((r - b) >= self.get_int('slow_red_margin_b'))
        )

        mask_u8 = (red_mask.astype(np.uint8) * 255)
        k = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)

        red_pixels = int(np.count_nonzero(mask_u8))
        total_pixels = max(1, int(mask_u8.size))
        ratio_x1000 = int(red_pixels * 1000 / total_pixels)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0.0
        best_box = None

        for c in contours:
            area = float(cv2.contourArea(c))
            if area > max_area:
                max_area = area
                bx, by, bw, bh = cv2.boundingRect(c)
                best_box = (bx, by, bw, bh)

        box_ok = False
        inner_ok = True
        white_ratio_x1000 = 0
        black_pixels = 0
        black_ratio_x1000 = 0

        if best_box is not None:
            bx, by, bw, bh = best_box
            box_ok = bw >= self.get_int('slow_min_box_w') and bh >= self.get_int('slow_min_box_h')

            # ROI 경계에 붙은 빨간 반사/잘린 물체는 서행 표지판으로 보지 않음
            edge = self.get_int('slow_reject_edge_px')
            roi_hh, roi_ww = roi.shape[:2]
            box_edge_ok = (
                bx > edge and by > edge and
                (bx + bw) < (roi_ww - edge) and
                (by + bh) < (roi_hh - edge)
            )
            box_ok = box_ok and box_edge_ok

            if self.get_bool('slow_inner_check_enable') and box_ok:
                margin = self.get_float('slow_inner_margin_ratio')
                mx = int(bw * margin)
                my = int(bh * margin)

                ix1 = int(self.clamp(bx + mx, 0, roi.shape[1] - 1))
                iy1 = int(self.clamp(by + my, 0, roi.shape[0] - 1))
                ix2 = int(self.clamp(bx + bw - mx, ix1 + 1, roi.shape[1]))
                iy2 = int(self.clamp(by + bh - my, iy1 + 1, roi.shape[0]))

                inner = roi[iy1:iy2, ix1:ix2]

                if inner.size == 0:
                    inner_ok = False
                else:
                    ib = inner[:, :, 0].astype(np.int16)
                    ig = inner[:, :, 1].astype(np.int16)
                    ir = inner[:, :, 2].astype(np.int16)

                    maxc = np.maximum(np.maximum(ir, ig), ib)
                    minc = np.minimum(np.minimum(ir, ig), ib)

                    white_mask = (
                        (ir >= self.get_int('slow_white_min')) &
                        (ig >= self.get_int('slow_white_min')) &
                        (ib >= self.get_int('slow_white_min')) &
                        ((maxc - minc) <= self.get_int('slow_white_delta_max'))
                    )

                    black_mask = (
                        (ir <= self.get_int('slow_black_max')) &
                        (ig <= self.get_int('slow_black_max')) &
                        (ib <= self.get_int('slow_black_max'))
                    )

                    inner_total = max(1, int(white_mask.size))
                    white_pixels = int(np.count_nonzero(white_mask))
                    black_pixels = int(np.count_nonzero(black_mask))

                    white_ratio_x1000 = int(white_pixels * 1000 / inner_total)
                    black_ratio_x1000 = int(black_pixels * 1000 / inner_total)

                    inner_ok = (
                        white_ratio_x1000 >= self.get_int('slow_min_white_ratio_x1000') and
                        black_pixels >= self.get_int('slow_min_black_pixels') and
                        black_ratio_x1000 >= self.get_int('slow_min_black_ratio_x1000')
                    )

        raw_detected = (
            red_pixels >= self.get_int('slow_min_pixels') and
            ratio_x1000 >= self.get_int('slow_min_ratio_x1000') and
            max_area >= self.get_float('slow_min_contour_area') and
            box_ok and
            inner_ok
        )

        if raw_detected:
            self.slow_red_count += 1
            self.slow_clear_count = 0
        else:
            self.slow_clear_count += 1
            self.slow_red_count = 0

        if self.slow_red_count >= self.get_int('slow_confirm_frames'):
            self.slow_sign_state = True
            self.slow_sign_until = now + self.get_float('slow_sign_hold_time')

        if self.slow_clear_count >= self.get_int('slow_clear_frames') and now > self.slow_sign_until:
            self.slow_sign_state = False

        active = self.slow_sign_state or now < self.slow_sign_until

        out = Bool()
        out.data = bool(active)
        self.slow_sign_pub.publish(out)

        if now - self.last_slow_log_time >= self.get_float('slow_sign_log_interval'):
            self.last_slow_log_time = now
            self.get_logger().warn(
                f'SLOW_SIGN active={active} raw={raw_detected} '
                f'red_pix={red_pixels} ratio={ratio_x1000}/1000 '
                f'area={max_area:.1f} box={best_box} '
                f'white={white_ratio_x1000}/1000 black={black_pixels}/{black_ratio_x1000}/1000'
            )

    def slow_sign_active(self, now):
        if not self.get_bool('slow_sign_enable'):
            return False
        return self.slow_sign_state or now < self.slow_sign_until


    def make_yellow_mask(self, roi):
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.get_int('h_min'), self.get_int('s_min'), self.get_int('v_min')], dtype=np.uint8)
        upper = np.array([self.get_int('h_max'), self.get_int('s_max'), self.get_int('v_max')], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        k     = np.ones((5, 5), np.uint8)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def apply_line_ignore_roi(self, mask, band_y1, band_y2, frame_h, frame_w):
        if not self.get_bool('line_ignore_yellow_signal_enable'):
            return mask

        ix1 = int(frame_w * self.get_float('line_ignore_left'))
        ix2 = int(frame_w * self.get_float('line_ignore_right'))
        iy1 = int(frame_h * self.get_float('line_ignore_top'))
        iy2 = int(frame_h * self.get_float('line_ignore_bottom'))

        ix1 = int(self.clamp(ix1, 0, frame_w - 1))
        ix2 = int(self.clamp(ix2, ix1 + 1, frame_w))
        iy1 = int(self.clamp(iy1, 0, frame_h - 1))
        iy2 = int(self.clamp(iy2, iy1 + 1, frame_h))

        oy1 = max(band_y1, iy1)
        oy2 = min(band_y2, iy2)

        if oy2 > oy1:
            mask[oy1 - band_y1:oy2 - band_y1, ix1:ix2] = 0

        return mask

    def detect_center_in_band(self, frame, y1, y2, min_area):
        h, w = frame.shape[:2]
        roi  = frame[y1:y2, :]
        mask = self.make_yellow_mask(roi)
        mask = self.apply_line_ignore_roi(mask, y1, y2, h, w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0.0, mask
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < min_area:
            return None, None, area, mask
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None, area, mask
        cx       = int(M['m10'] / M['m00'])
        cy_local = int(M['m01'] / M['m00'])
        return cx, y1 + cy_local, area, mask

    def apply_angular_slew(self, target_angular, dt):
        max_delta     = self.get_float('angular_slew_rate') * dt
        delta         = self.clamp(target_angular - self.last_angular, -max_delta, max_delta)
        out           = self.last_angular + delta
        self.last_angular = out
        return out

    def interpolate_x_at_distance(self, points, target_dist):
        if not points:
            return None
        points = sorted(points, key=lambda p: p[0])
        if target_dist <= points[0][0]:  return points[0][1]
        if target_dist >= points[-1][0]: return points[-1][1]
        for i in range(len(points) - 1):
            d0, x0 = points[i]
            d1, x1 = points[i + 1]
            if d0 <= target_dist <= d1:
                ratio = (target_dist - d0) / max(d1 - d0, 1e-6)
                return x0 + ratio * (x1 - x0)
        return points[-1][1]

    def detect_fitline_fallback(self, frame, h, w):
        y1   = int(h * self.get_float('fitline_y_start_ratio'))
        y2   = int(h * self.get_float('fitline_y_end_ratio'))
        roi  = frame[y1:y2, :]
        mask = self.make_yellow_mask(roi)
        mask = self.apply_line_ignore_roi(mask, y1, y2, h, w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, mask
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < self.get_float('fitline_min_area'):
            return None, area, mask
        pts = largest.reshape(-1, 2)
        if len(pts) < 5:
            M = cv2.moments(largest)
            if M['m00'] == 0:
                return None, area, mask
            cx = float(M['m10'] / M['m00'])
            return self.clamp(cx, 0.0, float(w - 1)), area, mask
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx[0]), float(vy[0])
        x0, y0 = float(x0[0]), float(y0[0])
        target_y_local = h * self.get_float('fitline_target_y_ratio') - y1
        if abs(vy) < 1e-3:
            target_x = x0
        else:
            target_x = x0 + (target_y_local - y0) * (vx / vy)
        return self.clamp(target_x, 0.0, float(w - 1)), area, mask

    def detect_vision_obstacle(self, frame):
        h, w = frame.shape[:2]
        y1 = int(h * self.get_float('vision_obstacle_y_start_ratio'))
        y2 = int(h * self.get_float('vision_obstacle_y_end_ratio'))
        center_width = self.get_float('vision_obstacle_center_width_ratio')
        x1 = int(w * (0.5 - center_width * 0.5))
        x2 = int(w * (0.5 + center_width * 0.5))
        y1 = self.clamp(y1, 0, h - 1)
        y2 = self.clamp(y2, y1 + 1, h)
        x1 = self.clamp(x1, 0, w - 1)
        x2 = self.clamp(x2, x1 + 1, w)
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return False, None, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        yellow = self.make_yellow_mask(roi)
        dark_mask = (V < self.get_int('vision_obstacle_dark_v_max')).astype(np.uint8) * 255
        sat_mask  = (
            (S > self.get_int('vision_obstacle_sat_min')) &
            (V > self.get_int('vision_obstacle_v_min'))
        ).astype(np.uint8) * 255
        obstacle_mask = cv2.bitwise_or(dark_mask, sat_mask)
        obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(yellow))
        k = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, k)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, 0.0
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < self.get_float('vision_obstacle_min_area'):
            return False, None, area
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return False, None, area
        cx_local = int(M['m10'] / M['m00'])
        cx_abs   = int(x1 + cx_local)
        return True, cx_abs, area

    def make_vision_avoid_twist(self, frame, now, cx_near):
        if not self.get_bool('vision_avoid_enable'):
            return None
        obstacle, obs_cx, obs_area = self.detect_vision_obstacle(frame)
        self.vision_obstacle_seen = obstacle
        self.vision_obstacle_cx   = obs_cx
        self.vision_obstacle_area = obs_area
        if self.vision_avoid_state == 'FOLLOW':
            if obstacle:
                self.vision_avoid_state      = 'TURN_LEFT'
                self.vision_avoid_state_time = now
                self.first_detection         = True
                self.last_angular            = 0.0
                self.get_logger().warn(f'VISION_AVOID_START area={obs_area:.0f} cx={obs_cx} dir=LEFT')
            else:
                return None
        twist = Twist()
        elapsed = now - self.vision_avoid_state_time
        if self.vision_avoid_state == 'TURN_LEFT':
            twist.linear.x  = 0.0
            twist.angular.z = self.get_float('vision_avoid_turn_speed')
            if elapsed >= self.get_float('vision_avoid_turn_time'):
                self.vision_avoid_state      = 'BYPASS_LEFT'
                self.vision_avoid_state_time = now
            self._log_vision_avoid(now, 'VISION_TURN_LEFT', twist, obstacle, obs_area, cx_near)
            return twist
        if self.vision_avoid_state == 'BYPASS_LEFT':
            twist.linear.x  = self.get_float('vision_avoid_forward_speed')
            twist.angular.z = self.get_float('vision_avoid_bypass_left_w') if obstacle else -self.get_float('vision_avoid_recover_w')
            min_t = self.get_float('vision_avoid_bypass_min_time')
            max_t = self.get_float('vision_avoid_bypass_max_time')
            if (elapsed >= min_t and not obstacle) or elapsed >= max_t:
                self.vision_avoid_state      = 'RECOVER_RIGHT'
                self.vision_avoid_state_time = now
            self._log_vision_avoid(now, 'VISION_BYPASS_LEFT', twist, obstacle, obs_area, cx_near)
            return twist
        if self.vision_avoid_state == 'RECOVER_RIGHT':
            twist.linear.x  = self.get_float('vision_avoid_forward_speed') * 0.90
            twist.angular.z = -self.get_float('vision_avoid_recover_w')
            if cx_near is not None or elapsed >= self.get_float('vision_avoid_recover_time'):
                self.vision_avoid_state = 'FOLLOW'
                self.first_detection    = True
                self.last_angular       = 0.0
                self.get_logger().warn('VISION_AVOID_END: return to line follow')
                return None
            self._log_vision_avoid(now, 'VISION_RECOVER_RIGHT', twist, obstacle, obs_area, cx_near)
            return twist
        self.vision_avoid_state = 'FOLLOW'
        return None

    def _log_vision_avoid(self, now, label, twist, obstacle, obs_area, cx_near):
        if now - self.last_vision_avoid_log_time >= self.get_float('vision_avoid_log_interval'):
            self.last_vision_avoid_log_time = now
            self.get_logger().warn(
                f'{label} obs={obstacle} area={obs_area:.0f} '
                f'near={cx_near} v={twist.linear.x:.3f} w={twist.angular.z:.3f}'
            )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            self.publish_stop()
            return

        h, w = frame.shape[:2]
        image_center_x = self.clamp(w / 2.0 + self.get_float('center_offset_px'), 0.0, float(w - 1))

        far_y1  = int(h * self.get_float('far_y_start_ratio'))
        far_y2  = int(h * self.get_float('far_y_end_ratio'))
        mid_y1  = int(h * self.get_float('mid_y_start_ratio'))
        mid_y2  = int(h * self.get_float('mid_y_end_ratio'))
        near_y1 = int(h * self.get_float('near_y_start_ratio'))
        near_y2 = int(h * self.get_float('near_y_end_ratio'))

        cx_far,  cy_far,  area_far,  mask_far  = self.detect_center_in_band(frame, far_y1,  far_y2,  self.get_float('min_area_far'))
        cx_mid,  cy_mid,  area_mid,  mask_mid  = self.detect_center_in_band(frame, mid_y1,  mid_y2,  self.get_float('min_area_mid'))
        cx_near, cy_near, area_near, mask_near = self.detect_center_in_band(frame, near_y1, near_y2, self.get_float('min_area_near'))

        # [FIX] near 극단값 필터
        _near_margin = 18
        if cx_near is not None:
            if cx_near < _near_margin or cx_near > w - _near_margin:
                cx_near = None
            else:
                self.last_cx_near = cx_near

        now = time.time()
        self.update_traffic_light_roi(frame, now)
        self.update_slow_sign_roi(frame, now)
        # [FIX] 첫 프레임 dt 폭발 방지
        if self.first_image:
            self.last_seen_time = now
            self.last_time      = now
            self.first_image    = False
            dt = 0.1
        else:
            dt = max(now - self.last_time, 1e-3)
            self.last_time = now

        vision_avoid_twist = self.make_vision_avoid_twist(frame, now, cx_near)
        if vision_avoid_twist is not None:
            self.cmd_pub.publish(vision_avoid_twist)
            return

        if cx_near is not None and cx_mid is not None:
            curve_px_near_mid = abs(float(cx_mid) - float(cx_near))
            curvature_norm    = min(curve_px_near_mid / max(image_center_x, 1.0), 1.0)
            self.last_curvature_norm = curvature_norm
        else:
            curvature_norm    = self.last_curvature_norm * self.get_float('curvature_decay')
            self.last_curvature_norm = curvature_norm
            curve_px_near_mid = curvature_norm * image_center_x

        far_valid = False
        if cx_far is not None:
            ref = cx_mid if cx_mid is not None else (cx_near if cx_near is not None else None)
            if ref is None:
                far_valid = True
            else:
                far_valid = abs(float(cx_far) - float(ref)) <= self.get_float('far_max_jump_px')
        if curve_px_near_mid >= self.get_float('far_ignore_curve_px'):
            far_valid = False
        if far_valid and self.last_cx_near is not None and cx_far is not None:
            if abs(float(cx_far) - float(self.last_cx_near)) > self.get_float('far_max_jump_px'):
                far_valid = False

        points = []
        if cx_near is not None:
            points.append((self.get_float('near_dist_m'), float(cx_near)))
        if cx_mid is not None:
            points.append((self.get_float('mid_dist_m'),  float(cx_mid)))
        if cx_far is not None and far_valid:
            points.append((self.get_float('far_dist_m'),  float(cx_far)))

        fallback_target_x, fallback_area, fallback_mask = self.detect_fitline_fallback(frame, h, w)

        twist = Twist()

        if points or fallback_target_x is not None:
            self.last_seen_time = now

            la_s = self.get_float('lookahead_straight_m')
            la_t = self.get_float('lookahead_turn_m')
            la_m = la_s - (la_s - la_t) * curvature_norm
            la_m = self.clamp(la_m, la_t, la_s)

            target_x = self.interpolate_x_at_distance(points, la_m) if points else None
            if target_x is None and points:
                target_x = points[0][1]

            fitline_used = False
            need_fitline = (fallback_target_x is not None and cx_near is None)

            if need_fitline:
                blend = self.get_float('fitline_blend_low') if len(points) >= 2 else self.get_float('fitline_blend')
                if target_x is None:
                    target_x = fallback_target_x
                else:
                    target_x = (1.0 - blend) * target_x + blend * fallback_target_x
                curvature_norm = max(curvature_norm, self.get_float('fitline_min_curvature_norm'))
                fitline_used = True

            if fitline_used and cx_near is None and cx_mid is None:
                if self.last_cx_near is not None:
                    if abs(target_x - float(self.last_cx_near)) > self.get_float('max_target_shift_px'):
                        target_x     = float(self.last_cx_near)
                        fitline_used = False

            if target_x is None:
                self.first_detection = True
                self.publish_stop()
                return

            # [NEW] predictive curve boost with straight gate
            # 코너에서는 선제 보정, 직선 진입 후에는 자동 OFF
            if self.get_bool('predictive_curve_enable'):
                pred_src = None

                # far까지 쓰면 코너 탈출 후에도 잔류 예측이 남아서 한쪽으로 끌 수 있음.
                # 그래서 예측 보정은 near-mid가 둘 다 보일 때만 사용.
                if cx_near is not None and cx_mid is not None:
                    pred_src = float(cx_mid) - float(cx_near)

                if pred_src is not None:
                    straight_px = self.get_float('predictive_curve_straight_px')
                    curv_min = self.get_float('predictive_curve_curv_min')

                    # near-mid 차이가 작거나 현재 곡률이 낮으면 직선으로 보고 예측 보정 OFF
                    if abs(pred_src) >= straight_px and curvature_norm >= curv_min:
                        if abs(pred_src) >= self.get_float('predictive_curve_min_px'):
                            pred_shift = self.get_float('predictive_curve_gain') * pred_src
                            pred_shift = self.clamp(
                                pred_shift,
                                -self.get_float('predictive_curve_max_px'),
                                self.get_float('predictive_curve_max_px')
                            )
                            target_x = self.clamp(float(target_x) + pred_shift, 0.0, float(w - 1))

            clamp_ref = cx_near if cx_near is not None else self.last_cx_near
            if clamp_ref is not None:
                ms       = self.get_float('max_target_shift_px')
                target_x = self.clamp(target_x, float(clamp_ref) - ms, float(clamp_ref) + ms)

            # [CORNER_EXIT_RECENTER] 코너 후 직진 진입에서 라인이 한쪽으로 크게 밀린 상황 감지
            corner_exit_recenter = False
            raw_target_x = float(target_x)

            if self.get_bool('corner_exit_recenter_enable'):
                preview_err = raw_target_x - image_center_x

                near_edge = False
                if cx_near is not None:
                    near_edge = abs(float(cx_near) - image_center_x) >= self.get_float('recenter_near_edge_px')

                corner_exit_recenter = (
                    curvature_norm <= self.get_float('recenter_curv_max') and
                    (
                        abs(preview_err) >= self.get_float('recenter_err_px') or
                        near_edge
                    )
                )

            # [NEW] target_x smoothing filter
            if self.filtered_target_x is None or self.first_detection:
                self.filtered_target_x = float(target_x)
            else:
                alpha = self.get_float('target_smooth_alpha')
                max_step = self.get_float('target_smooth_max_step_px')

                if corner_exit_recenter:
                    alpha = max(alpha, self.get_float('recenter_smooth_alpha'))
                    max_step = max(max_step, self.get_float('recenter_smooth_max_step_px'))

                desired = self.filtered_target_x + alpha * (float(target_x) - self.filtered_target_x)
                delta = self.clamp(desired - self.filtered_target_x, -max_step, max_step)
                self.filtered_target_x += delta
            target_x = self.filtered_target_x

            target_error = float(target_x - image_center_x)

            if self.first_detection:
                self.last_error    = target_error
                self.last_target_x = target_x
                self.first_detection = False

            derivative = (target_error - self.last_error) / dt
            self.last_error = target_error

            # [NEW] Feedforward: target 이동 속도 기반 선제 조향
            # 라인이 오른쪽으로 이동 중 → 미리 오른쪽으로 틀어줌 (도리도리 완화)
            target_vel = (target_x - self.last_target_x) / dt if self.last_target_x is not None else 0.0
            ff_angular = -self.get_float('kff') * target_vel
            self.last_target_x = target_x

            angular = -(self.get_float('kp_target') * target_error +
                        self.get_float('kd')        * derivative) + ff_angular

            # [NEW] straight tiny steering damping
            if curvature_norm < self.get_float('straight_curvature_thresh'):
                deadband = self.get_float('straight_deadband_px')
                if abs(target_error) < deadband:
                    angular = 0.0
                elif abs(target_error) < deadband * 2.0:
                    angular *= self.get_float('straight_angular_scale')

            if corner_exit_recenter:
                # err 음수면 라인이 왼쪽에 있음 → 왼쪽으로 강하게 복귀
                rec_sign = -1.0 if target_error > 0.0 else 1.0
                rec_abs = self.get_float('recenter_min_w')
                rec_abs += 0.003 * max(0.0, abs(target_error) - self.get_float('recenter_err_px'))
                rec_abs = self.clamp(rec_abs, self.get_float('recenter_min_w'), self.get_float('recenter_max_w'))

                if abs(angular) < rec_abs:
                    angular = rec_sign * rec_abs

            angular = self.clamp(angular, -self.get_float('max_angular'), self.get_float('max_angular'))
            if self.get_bool('invert_angular'):
                angular = -angular
            angular = self.apply_angular_slew(angular, dt)

            if   angular > 0: self.last_turn_sign =  1.0
            elif angular < 0: self.last_turn_sign = -1.0

            pos_norm  = min(abs(target_error) / max(image_center_x, 1.0), 1.0)
            edge_norm = 0.0
            if cx_near is not None:
                edge_norm = min(abs(float(cx_near) - image_center_x) / max(image_center_x, 1.0), 1.0)

            slowdown = (
                self.get_float('curve_slowdown') * max(curvature_norm, pos_norm) +
                self.get_float('edge_slowdown')  * edge_norm
            )
            slowdown = self.clamp(slowdown, 0.0, 0.85)

            linear = self.get_float('base_speed') * (1.0 - slowdown)
            linear = self.clamp(linear, self.get_float('min_speed'), self.get_float('max_speed'))

            if len(points) >= 2 and curvature_norm >= self.get_float('corner_speed_start_norm'):
                linear = max(linear, self.get_float('corner_speed_min'))
                linear = min(linear, self.get_float('max_speed'))

            if corner_exit_recenter:
                linear = min(linear, self.get_float('recenter_speed'))

            # [SLOW SIGN] speed cap
            if self.slow_sign_active(now):
                linear = min(linear, self.get_float('slow_sign_max_speed'))

            self.last_linear = linear
            twist.linear.x  = linear
            twist.angular.z = angular
            self.cmd_pub.publish(twist)

            if now - self.last_log_time >= self.get_float('log_interval'):
                self.last_log_time = now
                self.get_logger().info(
                    f'DIST near={cx_near} mid={cx_mid} far={cx_far} far_ok={far_valid} | '
                    f'la={la_m:.2f} tgt={target_x:.1f} err={target_error:.1f} '
                    f'curv={curvature_norm:.2f} fit={fitline_used} ff={ff_angular:.3f} | '
                    f'v={linear:.3f} w={angular:.3f}'
                )

        else:
            lost_time = now - self.last_seen_time
            self.first_detection = True
            self.last_target_x   = None
            self.filtered_target_x = None
            if lost_time < self.get_float('lost_timeout'):
                if lost_time < self.get_float('coast_time'):
                    # [NEW] 흰마커 통과: 멈추지 않고 마지막 방향 유지
                    twist.linear.x  = self.last_linear * self.get_float('coast_factor')
                    twist.angular.z = self.last_angular
                else:
                    twist.linear.x  = self.get_float('search_speed')
                    twist.angular.z = self.last_turn_sign * self.get_float('search_angular')
                self.cmd_pub.publish(twist)
                if now - self.last_log_time >= self.get_float('log_interval'):
                    self.last_log_time = now
                    mode = 'COAST' if lost_time < self.get_float('coast_time') else 'SEARCH'
                    self.get_logger().warn(f'TEMP_LOST[{mode}] {lost_time:.2f}s')
            else:
                self.publish_stop()
                self.last_angular = 0.0
                if now - self.last_log_time >= self.get_float('log_interval'):
                    self.last_log_time = now
                    self.get_logger().error('LINE LOST: STOP')

        if self.get_bool('publish_debug'):
            debug     = frame.copy()
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[far_y1:far_y2,   :] = mask_far
            full_mask[mid_y1:mid_y2,   :] = mask_mid
            full_mask[near_y1:near_y2, :] = mask_near
            cv2.rectangle(debug, (0, far_y1),  (w-1, far_y2),  (255,   0,   0), 1)
            cv2.rectangle(debug, (0, mid_y1),  (w-1, mid_y2),  (  0, 255,   0), 1)
            cv2.rectangle(debug, (0, near_y1), (w-1, near_y2), (  0,   0, 255), 1)
            cv2.line(debug, (int(image_center_x), 0), (int(image_center_x), h), (255, 255, 255), 1)
            if cx_far is not None:
                col = (255, 0, 0) if far_valid else (80, 80, 80)
                cv2.circle(debug, (cx_far,  cy_far),  6, col,          -1)
            if cx_mid is not None:
                cv2.circle(debug, (cx_mid,  cy_mid),  6, (0, 255,   0), -1)
            if cx_near is not None:
                cv2.circle(debug, (cx_near, cy_near), 6, (0,   0, 255), -1)
            try:
                if 'target_x' in locals() and target_x is not None:
                    cv2.circle(debug, (int(target_x), int(h * 0.92)), 7, (0, 255, 255), -1)
            except Exception:
                pass
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
                mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
                self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_bgr, encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f'debug publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok(): node.publish_stop()
        except Exception: pass
        try:
            node.destroy_node()
        except Exception: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()
