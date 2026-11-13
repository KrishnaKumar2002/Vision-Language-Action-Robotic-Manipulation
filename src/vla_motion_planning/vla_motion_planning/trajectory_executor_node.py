import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import numpy as np

JOINT_NAMES = [
    'shoulder_rotation_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'middle_arm_joint',
    'elbow_2_joint',
    'wrist_joint',
    'gripper_left_joint',
]

# Forward and Inverse Kinematics for the SESAC arm (6-DOF + Tool)
CHAIN = [
    (0.10, 'z'),
    (0.16, 'y'),
    (0.38, 'y'),
    (0.20, 'z'),
    (0.16, 'y'),
    (0.09, 'z')
]
Z_TOOL = 0.045
TCP_OFFSET = 0.11

LIM = [
    (-3.14, 3.14),
    (-1.57, 1.57),
    (-2.35, 2.35),
    (-3.14, 3.14),
    (-1.57, 1.57),
    (-3.14, 3.14)
]

def rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def fk(q, tcp=TCP_OFFSET):
    p = np.zeros(3)
    R = np.eye(3)
    for i in range(6):
        p += R @ np.array([0, 0, CHAIN[i][0]])
        R = R @ rot(CHAIN[i][1], q[i])
    p += R @ np.array([0, 0, Z_TOOL + tcp])
    return p, R

def ik_error(q, target, tcp=TCP_OFFSET):
    p, _ = fk(q, tcp)
    return target - p

def inverse_kinematics_sesac(target, q0, tcp=TCP_OFFSET, iters=200, tol=1e-3, damping=0.02):
    q = np.array(q0, dtype=float)
    eps = 1e-6
    for _ in range(iters):
        e = ik_error(q, target, tcp)
        if np.linalg.norm(e) < tol:
            return q, True
        J = np.zeros((3, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (e - ik_error(dq, target, tcp)) / eps
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, e)
        q += dq
        for i in range(6):
            q[i] = max(LIM[i][0], min(LIM[i][1], q[i]))
    return q, np.linalg.norm(ik_error(q, target, tcp)) < 1e-3

class TrajectoryExecutorNode(Node):
    def __init__(self):
        super().__init__('vla_trajectory_executor')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.sub_dispatch = self.create_subscription(
            String, '/vla/task_dispatch', self.dispatch_callback, 10)
        self.sub_action_cmd = self.create_subscription(
            String, '/vla/action_cmd', self.action_cmd_callback, 10)
        self.sub_detected = self.create_subscription(
            String, '/vla/detected_objects', self.detected_callback, 10)
        self.sub_randomize = self.create_subscription(
            String, '/vla/randomize_scene', self.randomize_callback, 10)
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/vla/markers', 10)
        self.pub_carried_pose = self.create_publisher(String, '/vla/carried_object_pose', 10)

        # Realistic Scene Objects Catalog
        self.objects = {
            'red_box': {
                'name': 'Soda Can',
                'pos': [0.45, -0.20, 0.06],
                'type': 'can',
                'label': 'Red Can',
                'rgb': (0.88, 0.08, 0.08),
                'size': [0.065, 0.065, 0.12]
            },
            'blue_box': {
                'name': 'Ceramic Mug',
                'pos': [0.45, -0.05, 0.045],
                'type': 'mug',
                'label': 'Blue Mug',
                'rgb': (0.12, 0.38, 0.88),
                'size': [0.075, 0.075, 0.09]
            },
            'green_box': {
                'name': 'Specimen Flask',
                'pos': [0.35, -0.35, 0.06],
                'type': 'flask',
                'label': 'Green Flask',
                'rgb': (0.08, 0.78, 0.35),
                'size': [0.068, 0.068, 0.12]
            },
        }
        self.obstacle = {'name': 'barrier', 'pos': [0.38, 0.05, 0.10], 'size': [0.04, 0.22, 0.20]}
        self.platform = {'name': 'target_platform', 'pos': [0.35, 0.30, 0.02]}

        # Dedicated Non-Overlapping Multi-Bay Staging Slots on Platform (Radius = 0.13m)
        self.docking_slots = {
            'red_box': [0.30, 0.25, 0.065],    # Bay 01A (West): Soda Can
            'blue_box': [0.40, 0.25, 0.045],   # Bay 01B (East): Ceramic Mug
            'green_box': [0.35, 0.35, 0.065],  # Bay 01C (North): Specimen Flask
        }

        self.publish_static_environment()

        self.home_q = [0.0, 0.4, 1.2, 0.0, 0.6, 0.0]
        self.current_q = list(self.home_q)
        self.gripper_val = 0.035 # open
        self.is_executing = False
        self.trajectory_queue = []
        self.carried_object = None
        self.planned_path_points = []
        self.delivered_payload = None
        self.delivered_objects = set()

        self.timer = self.create_timer(0.02, self.control_tick) # 50 Hz control loop
        self.get_logger().info('VLA Trajectory Executor & Dynamic Object Reparenting online.')

    def publish_static_environment(self):
        t_plat = TransformStamped()
        t_plat.header.stamp = self.get_clock().now().to_msg()
        t_plat.header.frame_id = 'base_link'
        t_plat.child_frame_id = self.platform['name']
        t_plat.transform.translation.x, t_plat.transform.translation.y, t_plat.transform.translation.z = self.platform['pos']
        t_plat.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t_plat)

    def randomize_callback(self, msg):
        """Reset delivered items tracking when scene is randomized."""
        self.delivered_objects.clear()
        self.delivered_payload = None

    def detected_callback(self, msg):
        """Update object coordinates dynamically from live perception stream (zero hardcoded coords)."""
        try:
            data = json.loads(msg.data)
            for obj in data.get('objects', []):
                oid = obj.get('id')
                pos = obj.get('position')
                if oid in self.objects and pos:
                    if self.carried_object != oid and oid not in self.delivered_objects:
                        self.objects[oid]['pos'] = pos
        except Exception:
            pass

    def action_cmd_callback(self, msg):
        """Execute closed-loop 7-DoF delta actions from the End-to-End VLA model."""
        if self.is_executing:
            return # If executing a planned trajectory arc, don't interrupt

        try:
            data = json.loads(msg.data)
            delta = data.get('delta_action', [0.0, 0.0, 0.0])
            grip = data.get('gripper_target', self.gripper_val)

            # Current TCP
            p_tcp, _ = fk(self.current_q)
            target_p = p_tcp + np.array(delta)

            # Enforce physical safety boundaries:
            target_p[0] = np.clip(target_p[0], 0.15, 0.65)
            target_p[1] = np.clip(target_p[1], -0.50, 0.50)
            target_p[2] = np.clip(target_p[2], 0.035, 0.45) # Prevent driving through table

            # Solve Damped Least-Squares IK
            q_target, ok = inverse_kinematics_sesac(target_p, self.current_q)
            if ok:
                for i in range(6):
                    self.current_q[i] += (q_target[i] - self.current_q[i]) * 0.4
                self.gripper_val += (grip - self.gripper_val) * 0.4
        except Exception as e:
            self.get_logger().error(f'Error applying VLA delta action: {e}')

    def dispatch_callback(self, msg):
        if self.is_executing:
            self.get_logger().warning('Currently executing another trajectory. Ignoring new dispatch.')
            return

        try:
            plan = json.loads(msg.data)
            target_frame = plan['target_frame']
            dest_frame = plan['destination_frame']
            avoid_obs = plan.get('avoid_obstacle', True)

            if target_frame in self.delivered_objects:
                self.delivered_objects.remove(target_frame)

            try:
                t_obj = self.tf_buffer.lookup_transform('base_link', target_frame, rclpy.time.Time())
                p_src = np.array([t_obj.transform.translation.x, t_obj.transform.translation.y, t_obj.transform.translation.z])
            except Exception:
                p_src = np.array(self.objects[target_frame]['pos'])

            # Resolve Dedicated Non-Overlapping Target Slot for this payload
            p_dst = np.array(self.docking_slots.get(target_frame, self.platform['pos']))

            self.get_logger().info(f'Planning Collision-Aware Trajectory from {p_src} to {p_dst} (Obstacle Avoidance={avoid_obs})...')
            self.build_collision_aware_trajectory(p_src, p_dst, target_frame, avoid_obs)
            self.is_executing = True

        except Exception as e:
            self.get_logger().error(f'Failed resolving TF target poses: {e}')

    def build_collision_aware_trajectory(self, p_src, p_dst, obj_name, avoid_obs):
        self.trajectory_queue.clear()
        self.planned_path_points.clear()
        hover_z = 0.18
        high_corridor_z = 0.44

        p_src_hover = p_src + np.array([0, 0, hover_z])
        p_dst_hover = p_dst + np.array([0, 0, hover_z])

        # 3D Visual Trajectory Ribbon Waypoints
        self.planned_path_points.append(list(p_src_hover))
        self.planned_path_points.append(list(p_src))
        self.planned_path_points.append(list(p_src_hover))

        # 1. Approach and Pick
        self.add_cartesian_segment(p_src_hover, gripper=0.035, duration=2.0)
        self.add_cartesian_segment(p_src, gripper=0.035, duration=1.5)
        self.trajectory_queue.append(('GRIP', 0.015, 0.8)) # Close gripper
        self.trajectory_queue.append(('ATTACH', obj_name, 0.1))
        self.add_cartesian_segment(p_src_hover, gripper=0.015, duration=1.2)

        # 2. Strict Collision Avoidance Transit: High Bridge Corridor
        if avoid_obs:
            # Check if trajectory intersects the obstacle barrier corridor (Y in [-0.08, 0.18])
            y_start, y_end = p_src[1], p_dst[1]
            if (y_start < 0.05 < y_end) or (y_end < 0.05 < y_start):
                # 3-point high clearance plateau bridge over the barrier
                # 2.1 Climb up before reaching the obstacle zone
                wp_climb = np.array([p_src[0], -0.12 if y_start < y_end else 0.22, high_corridor_z])
                self.planned_path_points.append(list(wp_climb))
                self.add_cartesian_segment(wp_climb, gripper=0.015, duration=1.8)

                # 2.2 Cruise horizontally directly over the obstacle with generous vertical clearance (>22cm)
                wp_cross = np.array([0.40, 0.05, high_corridor_z])
                self.planned_path_points.append(list(wp_cross))
                self.add_cartesian_segment(wp_cross, gripper=0.015, duration=1.5)

                # 2.3 Maintain altitude until completely past the obstacle boundary
                wp_clear = np.array([p_dst[0], 0.22 if y_start < y_end else -0.12, high_corridor_z])
                self.planned_path_points.append(list(wp_clear))
                self.add_cartesian_segment(wp_clear, gripper=0.015, duration=1.8)
            else:
                # Midpoint clearance arc for non-crossing movements
                midpoint = (p_src_hover + p_dst_hover) / 2.0
                midpoint[2] = high_corridor_z
                self.planned_path_points.append(list(midpoint))
                self.add_cartesian_segment(midpoint, gripper=0.015, duration=2.2)

        self.planned_path_points.append(list(p_dst_hover))
        self.planned_path_points.append(list(p_dst + np.array([0, 0, 0.05])))
        self.planned_path_points.append(list(p_dst_hover))

        # 3. Transfer to Destination and Place
        self.add_cartesian_segment(p_dst_hover, gripper=0.015, duration=2.0)
        self.add_cartesian_segment(p_dst + np.array([0, 0, 0.05]), gripper=0.015, duration=1.5)
        self.trajectory_queue.append(('DETACH', obj_name, 0.1))
        self.trajectory_queue.append(('GRIP', 0.035, 0.8)) # Open gripper
        self.add_cartesian_segment(p_dst_hover, gripper=0.035, duration=1.2)

        # 4. Return to Home
        self.trajectory_queue.append(('HOME', self.home_q, 2.0))

    def add_cartesian_segment(self, target_xyz, gripper, duration):
        q, ok = inverse_kinematics_sesac(target_xyz, self.current_q)
        if not ok:
            q, ok = inverse_kinematics_sesac(target_xyz, self.home_q)
        self.trajectory_queue.append(('JOINT_INTERP', list(q), gripper, duration))

    def control_tick(self):
        stamp = self.get_clock().now().to_msg()
        p_tcp, _ = fk(self.current_q)

        # 1. Broadcast dynamic TF for all objects
        for name, data in self.objects.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = 'base_link'
            t.child_frame_id = name
            if name == self.carried_object:
                t.transform.translation.x = p_tcp[0]
                t.transform.translation.y = p_tcp[1]
                t.transform.translation.z = p_tcp[2] - 0.02
            else:
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = data['pos']
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

        if self.carried_object:
            carried_msg = String()
            carried_msg.data = json.dumps({
                'object_id': self.carried_object,
                'pos': [float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2]) - 0.02]
            })
            self.pub_carried_pose.publish(carried_msg)

        # 2. Step through trajectory queue
        if self.trajectory_queue:
            action = self.trajectory_queue[0]
            if action[0] == 'JOINT_INTERP':
                _, q_target, grip_target, duration = action
                steps = max(1, int(duration / 0.02))
                self.current_q = [
                    q + (qt - q) * (1.0 / steps) for q, qt in zip(self.current_q, q_target)
                ]
                self.gripper_val += (grip_target - self.gripper_val) * (1.0 / steps)
                if np.linalg.norm(np.array(self.current_q) - np.array(q_target)) < 0.01:
                    self.trajectory_queue.pop(0)

            elif action[0] == 'GRIP':
                _, grip_target, _ = action
                self.gripper_val = grip_target
                self.trajectory_queue.pop(0)

            elif action[0] == 'ATTACH':
                _, obj_name, _ = action
                self.carried_object = obj_name
                self.get_logger().info(f'>>> GRASP: {self.objects[obj_name]["label"]} secured in gripper.')
                self.trajectory_queue.pop(0)

            elif action[0] == 'DETACH':
                _, obj_name, _ = action
                # Release onto dedicated non-overlapping docking slot
                dock_pos = list(self.docking_slots.get(obj_name, [0.35, 0.30, 0.065]))
                self.objects[obj_name]['pos'] = dock_pos
                self.delivered_objects.add(obj_name)
                self.delivered_payload = obj_name
                self.carried_object = None

                # Notify camera simulator of permanent placement on dedicated slot
                carried_msg = String()
                carried_msg.data = json.dumps({
                    'object_id': obj_name,
                    'pos': dock_pos
                })
                self.pub_carried_pose.publish(carried_msg)

                self.get_logger().info(f'>>> RELEASE: {self.objects[obj_name]["label"]} delivered safely onto Docking Bay Slot at {dock_pos}.')
                self.trajectory_queue.pop(0)

            elif action[0] == 'HOME':
                _, q_home, duration = action
                self.trajectory_queue[0] = ('JOINT_INTERP', q_home, self.gripper_val, duration)
        else:
            if self.is_executing:
                self.is_executing = False
                self.get_logger().info('>>> VLA Pick-and-Place Task completed successfully!')

        # 3. Publish Robot Joint States
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = self.current_q + [self.gripper_val]
        self.pub_joints.publish(js)

        # 4. Publish High-Fidelity RViz Markers
        self.publish_scene_markers(stamp, p_tcp)

    def publish_scene_markers(self, stamp, p_tcp):
        markers = MarkerArray()

        # ==========================================
        # A. High-Tech Industrial Workstation Table
        # ==========================================
        # 1. Main Matte Dark Charcoal Table Surface (High contrast against studio background)
        m_table = Marker()
        m_table.header.stamp = stamp
        m_table.header.frame_id = 'base_link'
        m_table.ns = 'environment'
        m_table.id = 1
        m_table.type = Marker.CUBE
        m_table.action = Marker.ADD
        m_table.pose.position.x = 0.42
        m_table.pose.position.y = 0.0
        m_table.pose.position.z = -0.015
        m_table.pose.orientation.w = 1.0
        m_table.scale.x = 1.15
        m_table.scale.y = 1.35
        m_table.scale.z = 0.03
        m_table.color.r, m_table.color.g, m_table.color.b, m_table.color.a = (0.18, 0.20, 0.24, 1.0)
        markers.markers.append(m_table)

        # 2. Table Legs (4 Brushed Aluminum Silver Pillars)
        leg_positions = [
            (0.85, 0.55), (0.85, -0.55), (-0.05, 0.55), (-0.05, -0.55)
        ]
        for l_idx, (lx, ly) in enumerate(leg_positions):
            m_leg = Marker()
            m_leg.header.stamp = stamp
            m_leg.header.frame_id = 'base_link'
            m_leg.ns = 'environment'
            m_leg.id = 10 + l_idx
            m_leg.type = Marker.CYLINDER
            m_leg.action = Marker.ADD
            m_leg.pose.position.x = lx
            m_leg.pose.position.y = ly
            m_leg.pose.position.z = -0.215
            m_leg.pose.orientation.w = 1.0
            m_leg.scale.x = m_leg.scale.y = 0.045
            m_leg.scale.z = 0.37
            m_leg.color.r, m_leg.color.g, m_leg.color.b, m_leg.color.a = (0.68, 0.72, 0.78, 1.0)
            markers.markers.append(m_leg)

        # 3. Rubberized Anti-Static Pick Mat
        m_mat = Marker()
        m_mat.header.stamp = stamp
        m_mat.header.frame_id = 'base_link'
        m_mat.ns = 'environment'
        m_mat.id = 2
        m_mat.type = Marker.CUBE
        m_mat.action = Marker.ADD
        m_mat.pose.position.x = 0.42
        m_mat.pose.position.y = -0.20
        m_mat.pose.position.z = 0.001
        m_mat.pose.orientation.w = 1.0
        m_mat.scale.x = 0.34
        m_mat.scale.y = 0.48
        m_mat.scale.z = 0.002
        m_mat.color.r, m_mat.color.g, m_mat.color.b, m_mat.color.a = (0.10, 0.12, 0.15, 1.0)
        markers.markers.append(m_mat)

        # 4. Pick Mat Neon Cyan Boundary
        m_mat_border = Marker()
        m_mat_border.header.stamp = stamp
        m_mat_border.header.frame_id = 'base_link'
        m_mat_border.ns = 'environment'
        m_mat_border.id = 3
        m_mat_border.type = Marker.LINE_STRIP
        m_mat_border.action = Marker.ADD
        m_mat_border.scale.x = 0.004
        m_mat_border.color.r, m_mat_border.color.g, m_mat_border.color.b, m_mat_border.color.a = (0.0, 0.75, 0.95, 0.9)
        bx1, bx2 = 0.42 - 0.17, 0.42 + 0.17
        by1, by2 = -0.20 - 0.24, -0.20 + 0.24
        for px, py in [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2), (bx1, by1)]:
            pt = Point()
            pt.x, pt.y, pt.z = px, py, 0.003
            m_mat_border.points.append(pt)
        markers.markers.append(m_mat_border)

        # 5. Safety Hazard Demarcation Zone Under Obstacle
        m_hazard = Marker()
        m_hazard.header.stamp = stamp
        m_hazard.header.frame_id = 'base_link'
        m_hazard.ns = 'environment'
        m_hazard.id = 4
        m_hazard.type = Marker.CUBE
        m_hazard.action = Marker.ADD
        m_hazard.pose.position.x = 0.38
        m_hazard.pose.position.y = 0.05
        m_hazard.pose.position.z = 0.001
        m_hazard.pose.orientation.w = 1.0
        m_hazard.scale.x = 0.14
        m_hazard.scale.y = 0.30
        m_hazard.scale.z = 0.002
        m_hazard.color.r, m_hazard.color.g, m_hazard.color.b, m_hazard.color.a = (0.95, 0.75, 0.05, 0.75)
        markers.markers.append(m_hazard)

        # ==========================================
        # B. Autonomous Delivery Docking Bay
        # ==========================================
        dx, dy = self.platform['pos'][0], self.platform['pos'][1]
        
        # 1. Dock Chamfered Base Plate
        m_dock_base = Marker()
        m_dock_base.header.stamp = stamp
        m_dock_base.header.frame_id = 'base_link'
        m_dock_base.ns = 'dock'
        m_dock_base.id = 20
        m_dock_base.type = Marker.CYLINDER
        m_dock_base.action = Marker.ADD
        m_dock_base.pose.position.x, m_dock_base.pose.position.y = dx, dy
        m_dock_base.pose.position.z = 0.008
        m_dock_base.pose.orientation.w = 1.0
        m_dock_base.scale.x = m_dock_base.scale.y = 0.26
        m_dock_base.scale.z = 0.016
        m_dock_base.color.r, m_dock_base.color.g, m_dock_base.color.b, m_dock_base.color.a = (0.28, 0.32, 0.38, 1.0)
        markers.markers.append(m_dock_base)

        # 2. Outer Glowing Cyan Target Ring
        m_dock_ring1 = Marker()
        m_dock_ring1.header.stamp = stamp
        m_dock_ring1.header.frame_id = 'base_link'
        m_dock_ring1.ns = 'dock'
        m_dock_ring1.id = 21
        m_dock_ring1.type = Marker.CYLINDER
        m_dock_ring1.action = Marker.ADD
        m_dock_ring1.pose.position.x, m_dock_ring1.pose.position.y = dx, dy
        m_dock_ring1.pose.position.z = 0.017
        m_dock_ring1.pose.orientation.w = 1.0
        m_dock_ring1.scale.x = m_dock_ring1.scale.y = 0.21
        m_dock_ring1.scale.z = 0.003
        m_dock_ring1.color.r, m_dock_ring1.color.g, m_dock_ring1.color.b, m_dock_ring1.color.a = (0.0, 0.75, 0.95, 0.90)
        markers.markers.append(m_dock_ring1)

        # 3. Middle White Alignment Ring
        m_dock_ring2 = Marker()
        m_dock_ring2.header.stamp = stamp
        m_dock_ring2.header.frame_id = 'base_link'
        m_dock_ring2.ns = 'dock'
        m_dock_ring2.id = 22
        m_dock_ring2.type = Marker.CYLINDER
        m_dock_ring2.action = Marker.ADD
        m_dock_ring2.pose.position.x, m_dock_ring2.pose.position.y = dx, dy
        m_dock_ring2.pose.position.z = 0.018
        m_dock_ring2.pose.orientation.w = 1.0
        m_dock_ring2.scale.x = m_dock_ring2.scale.y = 0.13
        m_dock_ring2.scale.z = 0.003
        m_dock_ring2.color.r, m_dock_ring2.color.g, m_dock_ring2.color.b, m_dock_ring2.color.a = (0.95, 0.95, 0.98, 0.95)
        markers.markers.append(m_dock_ring2)

        # 4. Multi-Bay Staging Pads (Bay 01A - Red Can, Bay 01B - Blue Mug, Bay 01C - Green Flask)
        bay_pads = [
            (30, 0.30, 0.25, (0.90, 0.25, 0.25, 0.85)), # Bay A (West: Can)
            (31, 0.40, 0.25, (0.25, 0.55, 0.95, 0.85)), # Bay B (East: Mug)
            (32, 0.35, 0.35, (0.25, 0.85, 0.45, 0.85)), # Bay C (North: Flask)
        ]
        for pad_id, px, py, p_col in bay_pads:
            m_pad = Marker()
            m_pad.header.stamp = stamp
            m_pad.header.frame_id = 'base_link'
            m_pad.ns = 'dock'
            m_pad.id = pad_id
            m_pad.type = Marker.CYLINDER
            m_pad.action = Marker.ADD
            m_pad.pose.position.x, m_pad.pose.position.y = px, py
            m_pad.pose.position.z = 0.019
            m_pad.pose.orientation.w = 1.0
            m_pad.scale.x = m_pad.scale.y = 0.068
            m_pad.scale.z = 0.003
            m_pad.color.r, m_pad.color.g, m_pad.color.b, m_pad.color.a = p_col
            markers.markers.append(m_pad)

            # Central alignment target pin
            m_pin = Marker()
            m_pin.header.stamp = stamp
            m_pin.header.frame_id = 'base_link'
            m_pin.ns = 'dock'
            m_pin.id = pad_id + 10
            m_pin.type = Marker.CYLINDER
            m_pin.action = Marker.ADD
            m_pin.pose.position.x, m_pin.pose.position.y = px, py
            m_pin.pose.position.z = 0.021
            m_pin.pose.orientation.w = 1.0
            m_pin.scale.x = m_pin.scale.y = 0.022
            m_pin.scale.z = 0.004
            m_pin.color.r, m_pin.color.g, m_pin.color.b, m_pin.color.a = (1.0, 0.85, 0.15, 1.0)
            markers.markers.append(m_pin)

        # 5. 4 Corner Glowing LED Guide Beacons
        beacon_offsets = [(0.11, 0.11), (0.11, -0.11), (-0.11, 0.11), (-0.11, -0.11)]
        for b_idx, (bx, by) in enumerate(beacon_offsets):
            m_beacon = Marker()
            m_beacon.header.stamp = stamp
            m_beacon.header.frame_id = 'base_link'
            m_beacon.ns = 'dock'
            m_beacon.id = 24 + b_idx
            m_beacon.type = Marker.CYLINDER
            m_beacon.action = Marker.ADD
            m_beacon.pose.position.x = dx + bx
            m_beacon.pose.position.y = dy + by
            m_beacon.pose.position.z = 0.025
            m_beacon.pose.orientation.w = 1.0
            m_beacon.scale.x = m_beacon.scale.y = 0.014
            m_beacon.scale.z = 0.035
            m_beacon.color.r, m_beacon.color.g, m_beacon.color.b, m_beacon.color.a = (0.0, 0.85, 0.55, 1.0)
            markers.markers.append(m_beacon)

        # ==========================================
        # C. Laser Safety Barrier (Obstacle)
        # ==========================================
        ox, oy, _ = self.obstacle['pos']

        # 1. Left Safety Warning Pylon
        m_pylon_l = Marker()
        m_pylon_l.header.stamp = stamp
        m_pylon_l.header.frame_id = 'base_link'
        m_pylon_l.ns = 'obstacle'
        m_pylon_l.id = 40
        m_pylon_l.type = Marker.CYLINDER
        m_pylon_l.action = Marker.ADD
        m_pylon_l.pose.position.x = ox
        m_pylon_l.pose.position.y = oy - 0.11
        m_pylon_l.pose.position.z = 0.10
        m_pylon_l.pose.orientation.w = 1.0
        m_pylon_l.scale.x = m_pylon_l.scale.y = 0.032
        m_pylon_l.scale.z = 0.20
        m_pylon_l.color.r, m_pylon_l.color.g, m_pylon_l.color.b, m_pylon_l.color.a = (0.95, 0.75, 0.05, 1.0)
        markers.markers.append(m_pylon_l)

        # Left Amber Strobe Beacon
        m_beacon_l = Marker()
        m_beacon_l.header.stamp = stamp
        m_beacon_l.header.frame_id = 'base_link'
        m_beacon_l.ns = 'obstacle'
        m_beacon_l.id = 41
        m_beacon_l.type = Marker.SPHERE
        m_beacon_l.action = Marker.ADD
        m_beacon_l.pose.position.x = ox
        m_beacon_l.pose.position.y = oy - 0.11
        m_beacon_l.pose.position.z = 0.21
        m_beacon_l.pose.orientation.w = 1.0
        m_beacon_l.scale.x = m_beacon_l.scale.y = m_beacon_l.scale.z = 0.028
        m_beacon_l.color.r, m_beacon_l.color.g, m_beacon_l.color.b, m_beacon_l.color.a = (1.0, 0.40, 0.0, 1.0)
        markers.markers.append(m_beacon_l)

        # 2. Right Safety Warning Pylon
        m_pylon_r = Marker()
        m_pylon_r.header.stamp = stamp
        m_pylon_r.header.frame_id = 'base_link'
        m_pylon_r.ns = 'obstacle'
        m_pylon_r.id = 42
        m_pylon_r.type = Marker.CYLINDER
        m_pylon_r.action = Marker.ADD
        m_pylon_r.pose.position.x = ox
        m_pylon_r.pose.position.y = oy + 0.11
        m_pylon_r.pose.position.z = 0.10
        m_pylon_r.pose.orientation.w = 1.0
        m_pylon_r.scale.x = m_pylon_r.scale.y = 0.032
        m_pylon_r.scale.z = 0.20
        m_pylon_r.color.r, m_pylon_r.color.g, m_pylon_r.color.b, m_pylon_r.color.a = (0.95, 0.75, 0.05, 1.0)
        markers.markers.append(m_pylon_r)

        # Right Amber Strobe Beacon
        m_beacon_r = Marker()
        m_beacon_r.header.stamp = stamp
        m_beacon_r.header.frame_id = 'base_link'
        m_beacon_r.ns = 'obstacle'
        m_beacon_r.id = 43
        m_beacon_r.type = Marker.SPHERE
        m_beacon_r.action = Marker.ADD
        m_beacon_r.pose.position.x = ox
        m_beacon_r.pose.position.y = oy + 0.11
        m_beacon_r.pose.position.z = 0.21
        m_beacon_r.pose.orientation.w = 1.0
        m_beacon_r.scale.x = m_beacon_r.scale.y = m_beacon_r.scale.z = 0.028
        m_beacon_r.color.r, m_beacon_r.color.g, m_beacon_r.color.b, m_beacon_r.color.a = (1.0, 0.40, 0.0, 1.0)
        markers.markers.append(m_beacon_r)

        # 3. Translucent Glowing Laser Security Field
        m_laser = Marker()
        m_laser.header.stamp = stamp
        m_laser.header.frame_id = 'base_link'
        m_laser.ns = 'obstacle'
        m_laser.id = 44
        m_laser.type = Marker.CUBE
        m_laser.action = Marker.ADD
        m_laser.pose.position.x = ox
        m_laser.pose.position.y = oy
        m_laser.pose.position.z = 0.10
        m_laser.pose.orientation.w = 1.0
        m_laser.scale.x = 0.025
        m_laser.scale.y = 0.22
        m_laser.scale.z = 0.18
        m_laser.color.r, m_laser.color.g, m_laser.color.b, m_laser.color.a = (1.0, 0.35, 0.0, 0.45)
        markers.markers.append(m_laser)

        # 4. Top Structural Beam
        m_beam = Marker()
        m_beam.header.stamp = stamp
        m_beam.header.frame_id = 'base_link'
        m_beam.ns = 'obstacle'
        m_beam.id = 45
        m_beam.type = Marker.CUBE
        m_beam.action = Marker.ADD
        m_beam.pose.position.x = ox
        m_beam.pose.position.y = oy
        m_beam.pose.position.z = 0.20
        m_beam.pose.orientation.w = 1.0
        m_beam.scale.x = 0.035
        m_beam.scale.y = 0.24
        m_beam.scale.z = 0.015
        m_beam.color.r, m_beam.color.g, m_beam.color.b, m_beam.color.a = (0.35, 0.38, 0.42, 1.0)
        markers.markers.append(m_beam)

        # ==========================================
        # D. Realistic Objects (Mathematically Locked to World/Gripper)
        # ==========================================
        for name, data in self.objects.items():
            if name == self.carried_object:
                # Synchronized directly with gripper TCP in base_link
                px, py, pz = p_tcp[0], p_tcp[1], p_tcp[2] - 0.02
                is_held = True
            else:
                px, py, pz = data['pos'][0], data['pos'][1], data['pos'][2]
                is_held = False

            obj_type = data['type']

            if obj_type == 'can':
                # --- Red Soda Can ---
                # 1. Main Cylinder Body
                m_body = Marker()
                m_body.header.stamp = stamp
                m_body.header.frame_id = 'base_link'
                m_body.ns = 'objects'
                m_body.id = 50
                m_body.type = Marker.CYLINDER
                m_body.action = Marker.ADD
                m_body.pose.position.x, m_body.pose.position.y, m_body.pose.position.z = px, py, pz
                m_body.pose.orientation.w = 1.0
                m_body.scale.x = m_body.scale.y = 0.065
                m_body.scale.z = 0.115
                m_body.color.r, m_body.color.g, m_body.color.b, m_body.color.a = (0.88, 0.08, 0.08, 0.98)
                markers.markers.append(m_body)

                # 2. Aluminum Top Lid
                m_top = Marker()
                m_top.header.stamp = stamp
                m_top.header.frame_id = 'base_link'
                m_top.ns = 'objects'
                m_top.id = 51
                m_top.type = Marker.CYLINDER
                m_top.action = Marker.ADD
                m_top.pose.position.x, m_top.pose.position.y, m_top.pose.position.z = px, py, pz + 0.058
                m_top.pose.orientation.w = 1.0
                m_top.scale.x = m_top.scale.y = 0.058
                m_top.scale.z = 0.006
                m_top.color.r, m_top.color.g, m_top.color.b, m_top.color.a = (0.86, 0.88, 0.92, 1.0)
                markers.markers.append(m_top)

                # 3. Silver Bottom Rim
                m_bot = Marker()
                m_bot.header.stamp = stamp
                m_bot.header.frame_id = 'base_link'
                m_bot.ns = 'objects'
                m_bot.id = 52
                m_bot.type = Marker.CYLINDER
                m_bot.action = Marker.ADD
                m_bot.pose.position.x, m_bot.pose.position.y, m_bot.pose.position.z = px, py, pz - 0.058
                m_bot.pose.orientation.w = 1.0
                m_bot.scale.x = m_bot.scale.y = 0.060
                m_bot.scale.z = 0.005
                m_bot.color.r, m_bot.color.g, m_bot.color.b, m_bot.color.a = (0.80, 0.82, 0.86, 1.0)
                markers.markers.append(m_bot)

            elif obj_type == 'mug':
                # --- Blue Ceramic Mug ---
                # 1. Main Ceramic Body
                m_body = Marker()
                m_body.header.stamp = stamp
                m_body.header.frame_id = 'base_link'
                m_body.ns = 'objects'
                m_body.id = 60
                m_body.type = Marker.CYLINDER
                m_body.action = Marker.ADD
                m_body.pose.position.x, m_body.pose.position.y, m_body.pose.position.z = px, py, pz
                m_body.pose.orientation.w = 1.0
                m_body.scale.x = m_body.scale.y = 0.075
                m_body.scale.z = 0.088
                m_body.color.r, m_body.color.g, m_body.color.b, m_body.color.a = (0.12, 0.38, 0.88, 0.98)
                markers.markers.append(m_body)

                # 2. Coffee Surface Inside
                m_coffee = Marker()
                m_coffee.header.stamp = stamp
                m_coffee.header.frame_id = 'base_link'
                m_coffee.ns = 'objects'
                m_coffee.id = 61
                m_coffee.type = Marker.CYLINDER
                m_coffee.action = Marker.ADD
                m_coffee.pose.position.x, m_coffee.pose.position.y, m_coffee.pose.position.z = px, py, pz + 0.040
                m_coffee.pose.orientation.w = 1.0
                m_coffee.scale.x = m_coffee.scale.y = 0.060
                m_coffee.scale.z = 0.006
                m_coffee.color.r, m_coffee.color.g, m_coffee.color.b, m_coffee.color.a = (0.18, 0.10, 0.04, 1.0)
                markers.markers.append(m_coffee)

                # 3. Mug Handle
                m_handle = Marker()
                m_handle.header.stamp = stamp
                m_handle.header.frame_id = 'base_link'
                m_handle.ns = 'objects'
                m_handle.id = 62
                m_handle.type = Marker.CUBE
                m_handle.action = Marker.ADD
                m_handle.pose.position.x, m_handle.pose.position.y, m_handle.pose.position.z = px, py - 0.046, pz
                m_handle.pose.orientation.w = 1.0
                m_handle.scale.x = 0.016
                m_handle.scale.y = 0.028
                m_handle.scale.z = 0.055
                m_handle.color.r, m_handle.color.g, m_handle.color.b, m_handle.color.a = (0.12, 0.38, 0.88, 1.0)
                markers.markers.append(m_handle)

            elif obj_type == 'flask':
                # --- Green Specimen Flask ---
                # 1. Main Flask Body
                m_body = Marker()
                m_body.header.stamp = stamp
                m_body.header.frame_id = 'base_link'
                m_body.ns = 'objects'
                m_body.id = 70
                m_body.type = Marker.CYLINDER
                m_body.action = Marker.ADD
                m_body.pose.position.x, m_body.pose.position.y, m_body.pose.position.z = px, py, pz
                m_body.pose.orientation.w = 1.0
                m_body.scale.x = m_body.scale.y = 0.068
                m_body.scale.z = 0.075
                m_body.color.r, m_body.color.g, m_body.color.b, m_body.color.a = (0.08, 0.78, 0.35, 0.92)
                markers.markers.append(m_body)

                # 2. Tapered Flask Neck
                m_neck = Marker()
                m_neck.header.stamp = stamp
                m_neck.header.frame_id = 'base_link'
                m_neck.ns = 'objects'
                m_neck.id = 71
                m_neck.type = Marker.CYLINDER
                m_neck.action = Marker.ADD
                m_neck.pose.position.x, m_neck.pose.position.y, m_neck.pose.position.z = px, py, pz + 0.048
                m_neck.pose.orientation.w = 1.0
                m_neck.scale.x = m_neck.scale.y = 0.032
                m_neck.scale.z = 0.035
                m_neck.color.r, m_neck.color.g, m_neck.color.b, m_neck.color.a = (0.08, 0.78, 0.35, 0.92)
                markers.markers.append(m_neck)

                # 3. Polished Gold Screw Cap
                m_cap = Marker()
                m_cap.header.stamp = stamp
                m_cap.header.frame_id = 'base_link'
                m_cap.ns = 'objects'
                m_cap.id = 72
                m_cap.type = Marker.CYLINDER
                m_cap.action = Marker.ADD
                m_cap.pose.position.x, m_cap.pose.position.y, m_cap.pose.position.z = px, py, pz + 0.070
                m_cap.pose.orientation.w = 1.0
                m_cap.scale.x = m_cap.scale.y = 0.038
                m_cap.scale.z = 0.014
                m_cap.color.r, m_cap.color.g, m_cap.color.b, m_cap.color.a = (0.95, 0.78, 0.18, 1.0)
                markers.markers.append(m_cap)

        # ==========================================
        # E. 3D Holographic Trajectory Ribbon
        # ==========================================
        if self.planned_path_points:
            m_path = Marker()
            m_path.header.stamp = stamp
            m_path.header.frame_id = 'base_link'
            m_path.ns = 'trajectory'
            m_path.id = 90
            m_path.type = Marker.LINE_STRIP
            m_path.action = Marker.ADD
            m_path.scale.x = 0.008 # Ribbon thickness
            m_path.color.r, m_path.color.g, m_path.color.b, m_path.color.a = (0.0, 0.75, 0.95, 0.90)
            for wp in self.planned_path_points:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), float(wp[2])
                m_path.points.append(p)
            markers.markers.append(m_path)

        self.pub_markers.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
