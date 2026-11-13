import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ObjectDetectorNode(Node):
    """Simulated 3D Perception System: Detects real-world objects and docking bays."""
    def __init__(self):
        super().__init__('vla_object_detector')
        self.pub_objects = self.create_publisher(String, '/vla/detected_objects', 10)

        # 3D Perception Catalog
        self.detected_catalog = [
            {
                'id': 'red_box',
                'name': 'can',
                'color': 'red',
                'label': '🥤 Red Soda Can (0.33L)',
                'frame_id': 'red_box',
                'position': [0.45, -0.20, 0.06]
            },
            {
                'id': 'blue_box',
                'name': 'mug',
                'color': 'blue',
                'label': '☕ Blue Ceramic Mug',
                'frame_id': 'blue_box',
                'position': [0.45, -0.05, 0.045]
            },
            {
                'id': 'green_box',
                'name': 'flask',
                'color': 'green',
                'label': '🧪 Green Specimen Flask',
                'frame_id': 'green_box',
                'position': [0.35, -0.35, 0.06]
            },
        ]

        self.timer = self.create_timer(0.5, self.publish_perception_stream) # 2 Hz stream
        self.get_logger().info('VLA 3D Object Perception Detector online. Streaming detected objects.')

    def publish_perception_stream(self):
        msg = String()
        msg.data = json.dumps({'objects': self.detected_catalog})
        self.pub_objects.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
