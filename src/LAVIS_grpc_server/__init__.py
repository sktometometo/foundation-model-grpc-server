import logging
from concurrent import futures

import cv2
import grpc
import torch
from lavis.models import load_model_and_preprocess

from LAVIS_grpc_server import lavis_server_pb2, lavis_server_pb2_grpc
from LAVIS_grpc_server.lavis_server_pb2_grpc import \
    add_LAVISServerServicer_to_server
from LAVIS_grpc_server.server import LAVISServer
from LAVIS_grpc_server.utils import cv_array_to_image_proto


def download_model_cache():
  logging.basicConfig(level=logging.INFO)
  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
  model, vis_processors, _ = load_model_and_preprocess(
      name="blip2_opt",
      model_type="pretrain_opt2.7b",
      is_eval=True,
      device=device)


def main_server():
  logging.basicConfig(level=logging.INFO)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  add_LAVISServerServicer_to_server(LAVISServer(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()


def main_client_sample():
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  cam = cv2.VideoCapture(0)
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = lavis_server_pb2_grpc.LAVISServerStub(channel)
    request = lavis_server_pb2.ImageCaptioningRequest()
    while True:
      ret, frame = cam.read()
      cv2.imshow('LAVISClient', frame)
      if cv2.waitKey(1) != -1:
        break
      image = cv_array_to_image_proto(frame)
      request.image.CopyFrom(image)
      result = stub.ImageCaptioning(request)
      logger.info('result: {}'.format(result))
  cv2.destroyAllWindows()
  logger.info('Finished')
