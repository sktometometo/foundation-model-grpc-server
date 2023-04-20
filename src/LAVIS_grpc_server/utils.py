import cv2
import numpy as np

from LAVIS_grpc_server import lavis_server_pb2


def image_proto_to_cv_array(image: lavis_server_pb2.Image) -> np.ndarray:
  """Convert Image proto to Opencv numpy array

  Args:
    image (lavis_server_pb2.Image): Image Proto
  
  Returns:
    cv_array (np.ndarray): RGB Image OpenCV array
  """
  # Seeing https://teratail.com/questions/203247
  cv_array = np.frombuffer(image.image_data,
                           dtype=np.uint8).reshape(image.height, image.width, 3)
  return cv_array


def cv_array_to_image_proto(cv_array: np.ndarray) -> lavis_server_pb2.Image:
  """Convert OpenCV numpy array to Image proto instance

  Args:
    cv_array (np.ndarray): RGB Image OpenCV array
  
  Returns:
    image (lavis_server_pb2.Image): Image Proto
  """
  height, width, channels = cv_array.shape[:3]
  image = lavis_server_pb2.Image()
  image.width = width
  image.height = height
  image.image_data = cv_array.tobytes()
  return image
