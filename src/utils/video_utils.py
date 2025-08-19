import cv2
import time
import numpy as np
from typing import Callable

def process_video_stream(source: int | str, frame_processor: Callable[[np.ndarray], None]):
    """
    Processes a video stream frame by frame.

    Args:
        source (int | str): The video source. 0 for default camera, or a path to a video file.
        frame_processor (Callable[[np.ndarray], None]): A function that takes a frame (np.ndarray) as input.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        frame_processor(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
