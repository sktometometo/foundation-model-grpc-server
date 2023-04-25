import argparse
import logging
from concurrent import futures

import cv2
import grpc

from LAVIS_grpc_server import lavis_server_pb2, lavis_server_pb2_grpc
from LAVIS_grpc_server.lavis_server_pb2_grpc import \
    add_LAVISServerServicer_to_server
from LAVIS_grpc_server.server import LAVISServer
from LAVIS_grpc_server.utils import (cv_array_to_image_proto,
                                     image_proto_to_cv_array)


def download_model_cache():
  logging.basicConfig(level=logging.INFO)
  from lavis.models import load_model_and_preprocess
  load_model_and_preprocess(name="blip2_opt",
                            model_type="pretrain_opt2.7b",
                            is_eval=True,
                            device='cpu')
  load_model_and_preprocess(name="blip2_t5",
                            model_type="pretrain_flant5xxl",
                            is_eval=True,
                            device='cpu')
  load_model_and_preprocess(name="blip_image_text_matching",
                            model_type="large",
                            is_eval=True,
                            device='cpu')
  load_model_and_preprocess(name="blip_vqa",
                            model_type="vqav2",
                            is_eval=True,
                            device='cpu')


def main_server():
  parser = argparse.ArgumentParser(description='LAVIS Server. \n' \
    + 'For image_captioning, blip2_opt with pretrain_opt2.7b is recommended.\n' \
    + 'For instructed_generation, blip2_t5 and pretrain_flant5xxl is recommended.\n' \
    + 'For text_localization, blip_image_text_matching and large is recommended.\n' \
    + 'For vqa, blip_vqa and vqav2 is recommended.' )
  parser.add_argument('--model-name', default='blip2_opt', help='model name')
  parser.add_argument('--model-type',
                      default='pretrain_opt2.7b',
                      help='model type')
  parser.add_argument('--port',
                      default=50051,
                      type=int,
                      help='Port for Grpc server')
  parser.add_argument('--use-gui',
                      action='store_true',
                      help='Show GUI Windows if set')
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  add_LAVISServerServicer_to_server(
      LAVISServer(use_gui=args.use_gui,
                  model_name=args.model_name,
                  model_type=args.model_type), server)
  server.add_insecure_port('[::]:{}'.format(args.port))
  server.start()
  server.wait_for_termination()


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
  parser.add_argument('--input-text',
                      default='I am a robot.',
                      type=str,
                      help='Input text for server if request has string field.')
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  cam = cv2.VideoCapture(0)
  with grpc.insecure_channel('{}:{}'.format(args.server_address,
                                            args.port)) as channel:
    stub = lavis_server_pb2_grpc.LAVISServerStub(channel)
    while True:
      ret, frame = cam.read()
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
  cv2.destroyAllWindows()
  logger.info('Finished')
