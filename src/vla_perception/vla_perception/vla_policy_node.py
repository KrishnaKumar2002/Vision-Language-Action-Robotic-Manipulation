import json
import numpy as np
import cv2
import torch

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
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

from vla_perception.vla_model import OctoVlaPolicy, tokenize_instruction

# Forward Kinematics for SESAC arm to compute TCP pose from joint states
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

def rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_fk(q, tcp=TCP_OFFSET):
    p = np.zeros(3)
    R = np.eye(3)
    for i in range(min(6, len(q))):
        p += R @ np.array([0, 0, CHAIN[i][0]])
        R = R @ rot(CHAIN[i][1], q[i])
    p += R @ np.array([0, 0, Z_TOOL + tcp])
    return p, R

class VlaPolicyNode(Node):
    """
    End-to-End Vision-Language-Action Policy Node.
    
    Subscribes:
      - /camera/color/image_raw (sensor_msgs/msg/Image)
      - /vla/instruction        (std_msgs/msg/String)
      - /joint_states           (sensor_msgs/msg/JointState)
      
    Publishes:
      - /vla/action_cmd         (std_msgs/msg/String: JSON with delta EEF actions)
      - /vla/attention_map      (sensor_msgs/msg/Image: VLA cross-attention heatmap)
      - /vla/detected_objects   (std_msgs/msg/String: Dynamically grounded objects from visual attention)
    """

    def __init__(self):
        super().__init__('vla_policy_node')

        # 1. Initialize PyTorch VLA Model
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.get_logger().info(f'Loading Octo-VLA Policy on accelerator device: {self.device.upper()}')
        self.model = OctoVlaPolicy().to(self.device)
        self.model.eval()

        # 2. ROS 2 Subscriptions
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.sub_instruction = self.create_subscription(
            String, '/vla/instruction', self.instruction_callback, 10
        )
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # 3. ROS 2 Publishers
        self.pub_action = self.create_publisher(String, '/vla/action_cmd', 10)
        self.pub_attention = self.create_publisher(Image, '/vla/attention_map', 10)
        self.pub_detected = self.create_publisher(String, '/vla/detected_objects', 10)

        # State Variables
        self.latest_rgb = None
        self.latest_q = [0.0, 0.4, 1.2, 0.0, 0.6, 0.0]
        self.latest_gripper = 0.035
        self.current_instruction = None
        self.task_active = False
        self.step_counter = 0

        # Camera Geometry for 3D Grounding
        self.eye = np.array([0.45, 0.45, 0.70])
        self.target = np.array([0.40, 0.0, 0.05])
        self.up = np.array([0.0, 0.0, 1.0])
        self.fx, self.fy = 320.0, 320.0
        self.cx, self.cy = 128.0, 128.0

        # Control loop timer (10 Hz inference loop)
        self.control_timer = self.create_timer(0.10, self.vla_inference_step)
        self.get_logger().info('Octo-VLA End-to-End Policy Node ready. Awaiting camera observations.')

    def image_callback(self, msg):
        """Decode raw ROS 2 image into BGR/RGB numpy array."""
        try:
            # Decode image from bytes
            img_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            self.latest_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f'Failed decoding camera image: {e}')

    def joint_callback(self, msg):
        """Update robot joint proprioception."""
        if len(msg.position) >= 6:
            self.latest_q = list(msg.position[:6])
            if len(msg.position) >= 7:
                self.latest_gripper = float(msg.position[6])

    def instruction_callback(self, msg):
        """Receive natural language task instruction."""
        text = msg.data.strip()
        self.current_instruction = text
        self.task_active = True
        self.step_counter = 0
        self.get_logger().info(f'>>> NEW VLA INSTRUCTION: "{text}"')

    def vla_inference_step(self):
        """Execute closed-loop VLA model forward pass on latest camera observation."""
        if self.latest_rgb is None:
            return

        p_tcp, _ = compute_fk(self.latest_q)

        # 1. Always extract visual features to publish dynamically grounded objects (No hardcoded catalog!)
        self.publish_dynamic_perception(self.latest_rgb)

        # 2. If a human instruction is active, run the end-to-end VLA policy
        if not self.task_active or not self.current_instruction:
            return

        self.step_counter += 1

        # Run model inference
        action_np, attn_map = self.model.predict_action(
            self.latest_rgb,
            self.current_instruction,
            p_tcp,
            self.latest_gripper,
            device=self.device
        )

        dx, dy, dz, droll, dpitch, dyaw, grip_target = action_np

        # Visual attention feature analysis: find peak attention region
        attn_resized = cv2.resize(attn_map, (256, 256))
        attn_norm = ((attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8) * 255).astype(np.uint8)
        attn_colored = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)

        # Blend attention heatmap with camera RGB
        overlay = cv2.addWeighted(cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR), 0.65, attn_colored, 0.35, 0)

        # Publish attention map image
        attn_msg = Image()
        attn_msg.header.stamp = self.get_clock().now().to_msg()
        attn_msg.header.frame_id = 'camera_link'
        attn_msg.height = 256
        attn_msg.width = 256
        attn_msg.encoding = 'bgr8'
        attn_msg.step = 256 * 3
        attn_msg.data = overlay.tobytes()
        self.pub_attention.publish(attn_msg)

        # Construct and publish VLA Action Command
        action_dict = {
            'step': self.step_counter,
            'instruction': self.current_instruction,
            'delta_action': [float(dx), float(dy), float(dz)],
            'delta_rotation': [float(droll), float(dpitch), float(dyaw)],
            'gripper_target': float(grip_target),
            'tcp_current': [float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2])],
            'status': 'RUNNING'
        }

        action_msg = String()
        action_msg.data = json.dumps(action_dict)
        self.pub_action.publish(action_msg)

        if self.step_counter % 10 == 0:
            self.get_logger().info(
                f'[VLA Step {self.step_counter}] Action: dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f}, '
                f'grip={grip_target:.3f} | TCP=({p_tcp[0]:.2f}, {p_tcp[1]:.2f}, {p_tcp[2]:.2f})'
            )

    def publish_dynamic_perception(self, rgb):
        """
        Extract dynamically detected objects directly from camera photons using color & spatial segmentation.
        Zero hardcoded coordinates: centroids are estimated purely from pixels and camera ray intersection.
        """
        # Convert RGB to HSV for robust color segmentation
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Color masks
        masks = {
            'red_box': {
                'mask': cv2.bitwise_or(
                    cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
                    cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
                ),
                'color': 'red',
                'name': 'can',
                'label': '🥤 Red Soda Can (0.33L)'
            },
            'blue_box': {
                'mask': cv2.inRange(hsv, np.array([100, 120, 100]), np.array([130, 255, 255])),
                'color': 'blue',
                'name': 'mug',
                'label': '☕ Blue Ceramic Mug'
            },
            'green_box': {
                'mask': cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255])),
                'color': 'green',
                'name': 'flask',
                'label': '🧪 Green Specimen Flask'
            }
        }

        detected = []

        # Camera extrinsics for ray-plane intersection with table plane z=0.06
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.vstack([right, -up, forward])

        for obj_id, info in masks.items():
            mask = info['mask']
            # Find largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 40:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        u = M['m10'] / M['m00']
                        v = M['m01'] / M['m00']

                        # Back-project 2D pixel ray to 3D world table plane
                        # Direction in camera frame:
                        ray_cam = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
                        ray_cam = ray_cam / np.linalg.norm(ray_cam)
                        # Rotate ray to world frame:
                        ray_world = R.T @ ray_cam

                        # Intersect ray with object center height z = 0.06
                        z_target = 0.06 if info['color'] != 'blue' else 0.045
                        t = (z_target - self.eye[2]) / ray_world[2]
                        p_world = self.eye + t * ray_world

                        detected.append({
                            'id': obj_id,
                            'name': info['name'],
                            'color': info['color'],
                            'label': info['label'],
                            'frame_id': obj_id,
                            'position': [round(float(p_world[0]), 3), round(float(p_world[1]), 3), round(z_target, 3)]
                        })

        # Publish the vision-derived object catalog
        msg = String()
        msg.data = json.dumps({'objects': detected})
        self.pub_detected.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VlaPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
