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
  if image.encoding == cv2.CV_8UC3:
    cv_array = np.frombuffer(image.image_data,
                             dtype=np.uint8).reshape(image.height, image.width,
                                                     3)
  elif image.encoding == cv2.CV_8UC3:
    cv_array = np.frombuffer(image.image_data,
                             dtype=np.uint8).reshape(image.height, image.width,
                                                     3)
  elif image.encoding == cv2.CV_32FC1:
    cv_array = np.frombuffer(image.image_data,
                             dtype=np.float32).reshape(image.height,
                                                       image.width, 1)
  else:
    raise ValueError('Unknown Image encoding: {}'.format(image.encoding))
  return cv_array


def cv_array_to_image_proto(cv_array: np.ndarray) -> lavis_server_pb2.Image:
  """Convert OpenCV numpy array to Image proto instance

  Args:
    cv_array (np.ndarray): RGB Image OpenCV array
  
  Returns:
    image (lavis_server_pb2.Image): Image Proto
  """
  dtype = cv_array.dtype
  height, width, channels = cv_array.shape[:3]
  image = lavis_server_pb2.Image()
  image.width = width
  image.height = height
  image.image_data = cv_array.tobytes()
  if dtype == np.uint8 and channels == 3:
    image.encoding = cv2.CV_8UC3
  elif dtype == np.uint8 and channels == 1:
    image.encoding = cv2.CV_8UC1
  elif dtype == np.float32 and channels == 1:
    image.encoding = cv2.CV_32FC1
  else:
    raise ValueError('Unknown OpenCV encoding. dtype: {}, channel: {}'.format(
        dtype, channels))
  return image
