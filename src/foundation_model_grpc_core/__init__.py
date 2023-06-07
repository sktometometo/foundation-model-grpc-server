import argparse
import logging

import cv2
import grpc

from foundation_model_grpc_interface import (lavis_server_pb2,
                                             lavis_server_pb2_grpc)
from foundation_model_grpc_utils import (cv_array_to_image_proto,
                                         image_proto_to_cv_array)


def main_client_sample():
  parser = argparse.ArgumentParser()
  parser.add_argument('--server-address',
                      default='localhost',
                      type=str,
                      help='Address for grpc server')
  parser.add_argument('--port',
                      default=50051,
                      type=int,
                      help='Port for grpc server')
  parser.add_argument(
      '--task',
      default='image_captioning',
      type=str,
      help=
      'Task type, options are \'image_captioning\', \'instructed_generation\', \'text_localization\', \'vqa\''
  )
  parser.add_argument('--use-gui', action='store_true', help='Use gui if set')
  parser.add_argument('--camera-id', default=0, type=int, help='Camera index')
  parser.add_argument('--once', action='store_true', help='Run once if set')
  parser.add_argument('--input-text',
                      default='I am a robot.',
                      type=str,
                      help='Input text for server if request has string field.')
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  cam = cv2.VideoCapture(args.camera_id)
  with grpc.insecure_channel('{}:{}'.format(args.server_address,
                                            args.port)) as channel:
    stub = lavis_server_pb2_grpc.LAVISServerStub(channel)
    while True:
      ret, frame = cam.read()
      if args.use_gui:
        cv2.imshow('LAVISClient', frame)
        if cv2.waitKey(1) != -1:
          break
      image = cv_array_to_image_proto(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      if args.task == 'image_captioning':
        request = lavis_server_pb2.ImageCaptioningRequest()
        request.image.CopyFrom(image)
        result = stub.ImageCaptioning(request, wait_for_ready=True)
        logger.info('caption: {}'.format(result.caption))
      elif args.task == 'instructed_generation':
        request = lavis_server_pb2.InstructedGenerationRequest()
        request.image.CopyFrom(image)
        request.prompt = args.input_text
        response = stub.InstructedGeneration(request, wait_for_ready=True)
        logger.info('response: {}'.format(response.response))
      elif args.task == 'text_localization':
        request = lavis_server_pb2.TextLocalizationRequest()
        request.image.CopyFrom(image)
        request.text = args.input_text
        result = stub.TextLocalization(request, wait_for_ready=True)
        attention_map = image_proto_to_cv_array(result.heatmap)
        logger.info('Attention_map: {}'.format(attention_map))
      elif args.task == 'vqa':
        request = lavis_server_pb2.VisualQuestionAnsweringRequest()
        request.image.CopyFrom(image)
        request.question = args.input_text
        result = stub.VisualQuestionAnswering(request, wait_for_ready=True)
        logger.info('response: {}'.format(result.answer))
      if args.once:
        break
  cv2.destroyAllWindows()
  logger.info('Finished')
