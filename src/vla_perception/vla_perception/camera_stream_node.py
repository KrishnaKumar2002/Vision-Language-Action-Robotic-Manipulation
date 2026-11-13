import math
import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String, Empty
    from geometry_msgs.msg import TransformStamped
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False
    class Node:
        def __init__(self, *args, **kwargs):
            pass
        def create_publisher(self, *args, **kwargs):
            return None
        def create_subscription(self, *args, **kwargs):
            return None
        def create_timer(self, *args, **kwargs):
            return None
        def get_logger(self):
            class DummyLogger:
                def info(self, msg): print(f"[INFO] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
                def warning(self, msg): print(f"[WARN] {msg}")
            return DummyLogger()
        def get_clock(self):
            class DummyClock:
                def now(self):
                    class DummyTime:
                        def to_msg(self): return None
                    return DummyTime()
            return DummyClock()

class CameraStreamNode(Node):
    """
    Simulated RGB-D Perception Sensor Node.
    Renders realistic synthetic camera images of the tabletop environment from the
    overhead camera perspective (camera_link) and publishes standard ROS 2 camera topics.
    
    Published Topics:
      - /camera/color/image_raw (sensor_msgs/msg/Image)
      - /camera/depth/image_raw (sensor_msgs/msg/Image)
      - /camera/camera_info     (sensor_msgs/msg/CameraInfo)
    
    Subscribed Topics:
      - /vla/randomize_scene    (std_msgs/msg/String or Empty) -> Repositions objects randomly
      - /vla/gripper_state      (std_msgs/msg/String) -> Updates carried object in camera view
    """

    def __init__(self):
        super().__init__('vla_camera_stream_node')

        # Camera Intrinsics
        self.width = 256
        self.height = 256
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 128.0
        self.cy = 128.0

        # Camera Extrinsics (Eye position overlooking workspace)
        # Eye at [0.45, 0.45, 0.70], looking at workbench center [0.40, 0.0, 0.05]
        self.eye = np.array([0.45, 0.45, 0.70])
        self.target = np.array([0.40, 0.0, 0.05])
        self.up = np.array([0.0, 0.0, 1.0])
        self._compute_view_matrix()

        # Publishers
        self.pub_rgb = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # Subscribers for dynamic scene interaction
        self.sub_randomize = self.create_subscription(
            String, '/vla/randomize_scene', self.randomize_callback, 10
        )
        self.sub_carried = self.create_subscription(
            String, '/vla/carried_object_pose', self.carried_callback, 10
        )

        # Physical Scene Objects State (Independent physical state in simulator)
        self.scene_objects = {
            'red_can': {
                'name': 'Soda Can',
                'color_bgr': (25, 25, 220), # Vibrant red
                'color_type': 'red',
                'pos': np.array([0.45, -0.20, 0.06]),
                'radius': 0.033,
                'height': 0.12
            },
            'blue_mug': {
                'name': 'Ceramic Mug',
                'color_bgr': (220, 95, 25), # Blue
                'color_type': 'blue',
                'pos': np.array([0.45, -0.05, 0.045]),
                'radius': 0.038,
                'height': 0.09
            },
            'green_flask': {
                'name': 'Specimen Flask',
                'color_bgr': (35, 200, 35), # Green
                'color_type': 'green',
                'pos': np.array([0.35, -0.35, 0.06]),
                'radius': 0.034,
                'height': 0.12
            }
        }

        # Static Environment Features
        self.platform_pos = np.array([0.35, 0.30, 0.02])
        self.barrier_pos = np.array([0.38, 0.05, 0.10])
        self.robot_tcp_pos = np.array([0.35, 0.0, 0.30])
        self.carried_object_id = None

        # Camera frame timer (15 Hz)
        self.timer = self.create_timer(1.0 / 15.0, self.render_and_publish_stream)
        self.get_logger().info('VLA Synthetic RGB-D Camera Stream online. Streaming @ 15 Hz.')

    def _compute_view_matrix(self):
        """Construct standard 3D camera look-at coordinate frame."""
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Rotation matrix from world to camera
        self.R_world_to_cam = np.vstack([right, -up, forward]) # Camera frame: X right, Y down, Z forward
        self.t_world_to_cam = -self.R_world_to_cam @ self.eye

    def project_point(self, p_world):
        """Project 3D world coordinate to 2D image plane (u, v) and depth Z."""
        p_cam = self.R_world_to_cam @ p_world + self.t_world_to_cam
        zc = p_cam[2]
        if zc <= 0.01:
            return None, None, zc
        u = int(self.cx + (self.fx * p_cam[0]) / zc)
        v = int(self.cy + (self.fy * p_cam[1]) / zc)
        return u, v, zc

    def randomize_callback(self, msg):
        """Randomize object positions dynamically to test that VLA does not use fixed coordinates."""
        rng = np.random.default_rng()
        # Randomize Red Can
        new_x = float(rng.uniform(0.38, 0.52))
        new_y = float(rng.uniform(-0.28, -0.12))
        self.scene_objects['red_can']['pos'] = np.array([new_x, new_y, 0.06])

        # Randomize Blue Mug
        new_x2 = float(rng.uniform(0.40, 0.50))
        new_y2 = float(rng.uniform(-0.08, 0.02))
        self.scene_objects['blue_mug']['pos'] = np.array([new_x2, new_y2, 0.045])

        self.get_logger().info(
            f'>>> RANDOMIZED SCENE: Red Can placed at ({new_x:.3f}, {new_y:.3f}), '
            f'Blue Mug at ({new_x2:.3f}, {new_y2:.3f})'
        )

    def carried_callback(self, msg):
        """Update position of object currently grasped by the robot."""
        import json
        try:
            data = json.loads(msg.data)
            obj_id = data.get('object_id')
            pos = data.get('pos')
            if obj_id in self.scene_objects and pos:
                self.scene_objects[obj_id]['pos'] = np.array(pos)
        except Exception as e:
            self.get_logger().error(f'Error updating carried object: {e}')

    def render_scene(self):
        """Renders 256x256 RGB and Depth image with perspective projection, shadows, and textures."""
        # 1. Background workbench: Dark titanium charcoal slate
        rgb = np.full((self.height, self.width, 3), (38, 42, 48), dtype=np.uint8)
        depth = np.full((self.height, self.width), 1.2, dtype=np.float32)

        # Draw anti-static ESD rubber mat with cyan border on table
        table_corners = [
            np.array([0.20, -0.45, 0.0]),
            np.array([0.65, -0.45, 0.0]),
            np.array([0.65, 0.45, 0.0]),
            np.array([0.20, 0.45, 0.0])
        ]
        pts_2d = []
        for p in table_corners:
            u, v, _ = self.project_point(p)
            if u is not None and 0 <= u < self.width and 0 <= v < self.height:
                pts_2d.append([u, v])
        if len(pts_2d) == 4:
            pts_np = np.array(pts_2d, dtype=np.int32)
            cv2.fillPoly(rgb, [pts_np], (48, 54, 62))
            cv2.polylines(rgb, [pts_np], isClosed=True, color=(180, 150, 40), thickness=2)

        # 2. Render Target Platform (Docking Bay 01): Concentric cyan, white, gold rings
        u_p, v_p, z_p = self.project_point(self.platform_pos)
        if u_p and 0 <= u_p < self.width and 0 <= v_p < self.height:
            r_pix = int((self.fx * 0.12) / z_p)
            cv2.circle(rgb, (u_p, v_p), r_pix, (90, 80, 70), -1) # base alloy deck
            cv2.circle(rgb, (u_p, v_p), int(r_pix * 0.8), (220, 180, 0), 2) # cyan ring
            cv2.circle(rgb, (u_p, v_p), int(r_pix * 0.5), (240, 240, 240), 2) # white ring
            cv2.circle(rgb, (u_p, v_p), int(r_pix * 0.2), (30, 200, 255), -1) # gold bullseye center

        # 3. Render Laser Obstacle Barrier
        u_b, v_b, z_b = self.project_point(self.barrier_pos)
        if u_b and 0 <= u_b < self.width and 0 <= v_b < self.height:
            w_pix = int((self.fx * 0.03) / z_b)
            h_pix = int((self.fy * 0.18) / z_b)
            cv2.rectangle(
                rgb,
                (u_b - w_pix // 2, v_b - h_pix // 2),
                (u_b + w_pix // 2, v_b + h_pix // 2),
                (30, 30, 240), # Glowing red laser barrier
                -1
            )
            cv2.rectangle(
                rgb,
                (u_b - w_pix // 2, v_b - h_pix // 2),
                (u_b + w_pix // 2, v_b + h_pix // 2),
                (200, 200, 255),
                1
            )

        # 4. Render Physical Objects (Red Can, Blue Mug, Green Flask)
        for obj_id, obj in self.scene_objects.items():
            pos = obj['pos']
            u_base, v_base, z_base = self.project_point(pos)
            u_top, v_top, z_top = self.project_point(pos + np.array([0, 0, obj['height']]))

            if u_base and 0 <= u_base < self.width and 0 <= v_base < self.height:
                r_pix = int((self.fx * obj['radius']) / z_base)
                # Shadow on table
                cv2.ellipse(rgb, (u_base + 3, v_base + 3), (r_pix + 2, int(r_pix * 0.6)), 0, 0, 360, (20, 22, 25), -1)

                # Cylinder Body
                color = obj['color_bgr']
                body_h = abs(v_base - v_top) if v_top else int((self.fy * obj['height']) / z_base)
                cv2.rectangle(
                    rgb,
                    (u_base - r_pix, v_base - body_h),
                    (u_base + r_pix, v_base),
                    color,
                    -1
                )
                # Shading / highlight
                cv2.line(
                    rgb,
                    (u_base - r_pix // 3, v_base - body_h),
                    (u_base - r_pix // 3, v_base),
                    (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60)),
                    2
                )
                # Lid / Top Rim (Silver for can, colored for mug/flask)
                lid_color = (200, 200, 200) if obj['color_type'] == 'red' else color
                cv2.ellipse(rgb, (u_base, v_base - body_h), (r_pix, int(r_pix * 0.4)), 0, 0, 360, lid_color, -1)

                # Depth buffer update
                cv2.circle(depth, (u_base, v_base - body_h // 2), r_pix, float(z_base), -1)

        # 5. Add slight camera sensor noise for realism
        noise = np.random.normal(0, 1.5, (self.height, self.width, 3)).astype(np.int16)
        rgb_noisy = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return rgb_noisy, depth

    def render_and_publish_stream(self):
        """Generate and publish RGB, Depth, and CameraInfo messages."""
        rgb, depth = self.render_scene()
        stamp = self.get_clock().now().to_msg()

        # 1. Publish RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = 'camera_link'
        rgb_msg.height = self.height
        rgb_msg.width = self.width
        rgb_msg.encoding = 'bgr8'
        rgb_msg.is_bigendian = 0
        rgb_msg.step = self.width * 3
        rgb_msg.data = rgb.tobytes()
        self.pub_rgb.publish(rgb_msg)

        # 2. Publish Depth Image (32FC1 in meters)
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'camera_link'
        depth_msg.height = self.height
        depth_msg.width = self.width
        depth_msg.encoding = '32FC1'
        depth_msg.is_bigendian = 0
        depth_msg.step = self.width * 4
        depth_msg.data = depth.tobytes()
        self.pub_depth.publish(depth_msg)

        # 3. Publish Camera Info
        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = 'camera_link'
        info_msg.height = self.height
        info_msg.width = self.width
        info_msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]
        info_msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        self.pub_info.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
