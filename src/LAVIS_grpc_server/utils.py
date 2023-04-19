import cv2
import numpy as np

from LAVIS_grpc_server.lavis_server_pb2 import Image


def image_proto_to_cv_array(image: Image):
  # Seeing https://teratail.com/questions/203247
  cv_array = np.frombuffer(image.image_data,
                           dtype=np.uint8).reshape(image.width, image.height, 3)
  return cv_array


def cv_array_to_image_proto(cv_array: np.ndarray):
  height, width, channels = cv_array.shape[:3]
  return Image()
