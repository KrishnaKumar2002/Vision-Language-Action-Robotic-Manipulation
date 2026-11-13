import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class LanguageTaskNode(Node):
    def __init__(self):
        super().__init__('vla_language_task_planner')
        self.sub_instruction = self.create_subscription(
            String, '/vla/instruction', self.instruction_callback, 10)
        self.sub_detected = self.create_subscription(
            String, '/vla/detected_objects', self.detected_callback, 10)
        
        self.pub_dispatch = self.create_publisher(String, '/vla/task_dispatch', 10)
        self.latest_objects = []

        self.get_logger().info('VLA Language Task Planner ready. Awaiting instructions on /vla/instruction.')

    def detected_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_objects = data.get('objects', [])
        except Exception as e:
            self.get_logger().error(f'Failed parsing detected objects: {e}')

    def instruction_callback(self, msg):
        text = msg.data.strip()
        self.get_logger().info(f'Received Language Instruction: "{text}"')
        self.parse_and_dispatch(text)

    def parse_and_dispatch(self, text):
        text_lower = text.lower()
        target_color = None

        # Multi-modal vocabulary grounding: colors & real-world object nouns
        if any(w in text_lower for w in ['red', 'can', 'coke', 'soda']):
            target_color = 'red'
        elif any(w in text_lower for w in ['blue', 'mug', 'coffee', 'cup']):
            target_color = 'blue'
        elif any(w in text_lower for w in ['green', 'flask', 'bottle', 'specimen']):
            target_color = 'green'

        if not target_color:
            self.get_logger().warning(f'Could not extract target object or color from instruction: "{text}"')
            return

        # Grounding: find matching object in visual perception stream
        matched_object = next((obj for obj in self.latest_objects if obj['color'] == target_color), None)
        if not matched_object:
            self.get_logger().error(f'Perception cannot find any "{target_color}" object in the scene!')
            return

        avoid_obstacle = any(w in text_lower for w in ['avoid', 'obstacle', 'barrier', 'laser', 'clear', 'over'])
        dest_frame = 'target_platform'

        plan = {
            'action': 'pick_and_place',
            'target_frame': matched_object['frame_id'],
            'target_name': matched_object.get('label', target_color),
            'target_color': target_color,
            'destination_frame': dest_frame,
            'avoid_obstacle': avoid_obstacle,
            'instruction': text
        }

        self.get_logger().info(
            f'Parsed VLA Plan: Target={matched_object["frame_id"]} ({matched_object.get("label", target_color)}) '
            f'-> Destination={dest_frame} (Avoid Obstacle={avoid_obstacle})'
        )
        dispatch_msg = String()
        dispatch_msg.data = json.dumps(plan)
        self.pub_dispatch.publish(dispatch_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LanguageTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
