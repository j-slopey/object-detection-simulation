import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import math
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)
model = YOLO("yolov8x-worldv2.pt")
# model.set_classes(["person"])

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
            results = model(frame)
            boxes = results[0].boxes

            # Uncomment the following to draw bounding boxes
            frame = results[0].plot()

            # Draw a target at the bottom of each bounding box
            for box in boxes:
                xywh = box.xywh.cpu()
                xywh = xywh.numpy()[0]
                # Center x
                box_mid_x = round(xywh[0])
                # Center y - height/2
                box_bottom_y = math.floor(xywh[1] + (xywh[3]/2))     

                # Be aware that y comes before x when setting pixels on a frame (hence frame[y][x])
                # Center
                frame[box_bottom_y][box_mid_x]  = [0, 0, 255]
                # Surrounding pixels
                frame[box_bottom_y + 1][box_mid_x + 1] = [255, 255, 0]
                frame[box_bottom_y + 1][box_mid_x - 1] = [255, 255, 0]
                frame[box_bottom_y + 1][box_mid_x] = [255, 255, 0]
                frame[box_bottom_y][box_mid_x - 1] = [255, 255, 0]
                frame[box_bottom_y][box_mid_x + 1] = [255, 255, 0]
                frame[box_bottom_y - 1][box_mid_x - 1] = [255, 255, 0]
                frame[box_bottom_y - 1][box_mid_x + 1] = [255, 255, 0]
                frame[box_bottom_y - 1][box_mid_x] = [255, 255, 0]
            
            self.latest_detection_frame = frame           

        else:
            self.stop()

    def stop(self):
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    cv2.namedWindow('Object Detection', cv2.WINDOW_AUTOSIZE)

    try:
        while image_subscriber.running:
            rclpy.spin_once(image_subscriber, timeout_sec=1)
            if image_subscriber.latest_detection_frame is not None:
                cv2.imshow('Object Detection', image_subscriber.latest_detection_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
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
        image_subscriber.destroy_node()
        cv2.destroyAllWindows()            
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()