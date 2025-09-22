"""
This module provides utility functions for processing video streams.

It includes a function to capture and process frames from a video file or a live camera feed.
"""

import cv2
import time
import numpy as np
from typing import Callable

def process_video_stream(source: int | str, frame_processor: Callable[[np.ndarray], None]):
    """
    Captures and processes frames from a video stream (file or camera) frame by frame.

    The processing of each frame is delegated to a provided `frame_processor` function.
    The stream continues until the end of the video or until the 'q' key is pressed.

    Args:
        source (int | str): The video source. Can be an integer (e.g., 0 for default camera)
                            or a string path to a video file.
        frame_processor (Callable[[np.ndarray], None]): A callable function that takes a single
                                                        NumPy array (representing a video frame)
                                                        as input and performs desired operations.
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
