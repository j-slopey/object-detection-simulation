import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import torch



device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.getLogger("ultralytics").setLevel(logging.WARNING)
model = YOLO("yolov8x-worldv2.pt")
model.to(device)


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        # Define subscriber for camera vision
        self.raw_image_subscription = self.create_subscription(
            Image, 
            '/camera_image',  # Topic name for simulated cam
            self.image_detection, 
            10)

        # Prevent unused variable warning
        self.raw_image_subscription
        self.running = True
        self.latest_detection_frame = None

    def image_detection(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame is not None:
            
            # Perform object detection with YOLO
            with torch.no_grad():
                results = model(frame)
                detection = results[0].plot()  # Plot detection boxes on the frame
                self.latest_detection_frame = detection
            
            # Draw dots at the center of detected bounding boxes (may be useful later for tracking detected objects?)
            boxes = results[0].boxes
            object_x, object_y = [None] * len(boxes), [None] * len(boxes)
            for i in range(len(boxes)):
                coord = boxes[i].xyxy
                object_x[i] = (coord[0][2].item() + coord[0][0].item()) / 2
                object_y[i] = (coord[0][3].item() + coord[0][1].item()) / 2

        else:
            self.stop()

    def stop(self):
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    cv2.namedWindow('Object Detection and Tracking', cv2.WINDOW_AUTOSIZE)

    try:
        while image_subscriber.running and rclpy.ok():
            rclpy.spin_once(image_subscriber, timeout_sec=1)
            if image_subscriber.latest_detection_frame is not None:
                cv2.imshow('Object Detection and Tracking', image_subscriber.latest_detection_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Object Detection and Tracking', cv2.WND_PROP_VISIBLE) < 1:
                image_subscriber.get_logger().info("Window Closed")
                image_subscriber.stop()
                break

    except KeyboardInterrupt:
        image_subscriber.get_logger().info("KeyboardInterrupt received, initiating shutdown.")
        image_subscriber.stop()
    except Exception as e:
        image_subscriber.get_logger().error(f"Unhandled exception in main loop: {e}")
        image_subscriber.stop()
    finally:
        image_subscriber.get_logger().info("Exiting main loop, cleaning up...")
        cv2.destroyAllWindows()
        if rclpy.ok(): # Check if context is still valid
            image_subscriber.destroy_node()
        if rclpy.ok_global_context(): # Check if global context is still valid for shutdown
            rclpy.try_shutdown()
        image_subscriber.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()