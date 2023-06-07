import argparse
import datetime
import logging
import os
from concurrent import futures

import cv2
import deep_translator
import grpc
import numpy as np
import torch
import yaml
from deep_translator import GoogleTranslator
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from PIL import Image

from foundation_model_grpc_core.lavis_server import LAVISServer
from foundation_model_grpc_interface import (lavis_server_pb2,
                                             lavis_server_pb2_grpc)
from foundation_model_grpc_interface.lavis_server_pb2 import (
    ImageCaptioningResponse, InstructedGenerationResponse,
    TextLocalizationResponse, VisualQuestionAnsweringResponse)
from foundation_model_grpc_interface.lavis_server_pb2_grpc import (
    LAVISServerServicer, add_LAVISServerServicer_to_server)
from foundation_model_grpc_utils import (cv_array_to_image_proto,
                                         image_proto_to_cv_array)

logger = logging.getLogger(__name__)


model_list = {
    'blip2_opt2.7b': {
        'model_name': 'blip2_opt',
        'model_type': 'pretrain_opt2.7b'
    },
    'blip2_flant5xl': {
        'model_name': 'blip2_t5',
        'model_type': 'pretrain_flant5xl'
    },
    'blip_text_localization': {
        'model_name': 'blip_image_text_matching',
        'model_type': 'large'
    },
    'blip_vqa': {
        'model_name': 'blip_vqa',
        'model_type': 'vqav2'
    }
}


class LAVISServer(LAVISServerServicer):

  def __init__(self,
               use_gui: bool,
               log_directory=None,
               model_device_dict={},
               use_translator=False,
               target_language='ja'):
    self.log_directory = log_directory
    self.use_gui = use_gui
    self.use_translator = use_translator
    self.input_translator = GoogleTranslator(source='auto', target='en')
    self.output_translator = GoogleTranslator(source='auto',
                                              target=target_language)

    self.models = {}
    for task_name, model_and_device in model_device_dict.items():
      device = torch.device(model_and_device['device'])
      model_name = model_and_device['model_name']
      model_type = model_and_device['model_type']
      model, vis_processors, text_processors = load_model_and_preprocess(
          name=model_name, model_type=model_type, is_eval=True, device=device)
      self.models[task_name] = {
          'model': model,
          'device': device,
          'vis_processors': vis_processors,
          'text_processors': text_processors
      }

    logger.info('Initialized')

  def translate_input_text(self, text):
    if self.use_translator:
      try:
        logger.info('original input text: {}'.format(text))
        return self.input_translator.translate(text)
      except deep_translator.exceptions.NotValidPayload as error:
        logger.error('Error: {}'.format(error))
        return text
    else:
      return text

  def translate_output_text(self, text):
    if self.use_translator:
      try:
        logger.info('original output text: {}'.format(text))
        return self.output_translator.translate(text)
      except deep_translator.exceptions.NotValidPayload as error:
        logger.error('Error: {}'.format(error))
        return text
    else:
      return text

  def __del__(self):
    if self.use_gui:
      cv2.destroyAllWindows()

  def save_data(self, path: str, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if type(data) == np.ndarray:
      cv2.imwrite(path, data)
    else:
      with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, encoding='utf-8', allow_unicode=True)

  def ImageCaptioning(self, request, context):
    # Retrieve models
    model = self.models['image_captioning']['model']
    device = self.models['image_captioning']['device']
    vis_processors = self.models['image_captioning']['vis_processors']
    text_processors = self.models['image_captioning']['text_processors']
    # Image
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv_array_rgb)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # Generate result
    result = model.generate({"image": image})
    raw_caption = result[0]
    caption = self.translate_output_text(raw_caption)
    #
    response = ImageCaptioningResponse(
        caption=caption)
    logger.info("Got image shape: {}".format(
        image_proto_to_cv_array(request.image).shape))
    logger.info("Generate caption: {} translated to {}".format(raw_caption, caption))
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory +
          '/{}_image_captioning_input.png'.format(current_datetime), cv_array_bgr)
      cv2.putText(cv_array_bgr, f'caption: {raw_caption}', (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
      self.save_data(
          self.log_directory + '/{}_image_captioning.png'.format(current_datetime),
          cv_array_bgr)
      self.save_data(
          self.log_directory +
          '/{}_image_captioning_input.yaml'.format(current_datetime),
          {'caption': caption,
           'raw_caption': raw_caption})
    if self.use_gui:
      cv2.imshow("LAVISBLIP2Server Image Captioning", cv_array_bgr)
      cv2.waitKey(1)
    return response

  def InstructedGeneration(self, request, context):
    model = self.models['instructed_generation']['model']
    device = self.models['instructed_generation']['device']
    vis_processors = self.models['instructed_generation']['vis_processors']
    text_processors = self.models['instructed_generation']['text_processors']
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    prompt = self.translate_input_text(request.prompt)
    raw_image = Image.fromarray(cv_array_rgb)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.generate({'image': image, 'prompt': prompt})
    response = InstructedGenerationResponse(
        response=self.translate_output_text(result[0]))
    logger.info("Get image with {} size with prompt {}".format(
        image_proto_to_cv_array(request.image).shape, prompt))
    logger.info("Generate caption: {}".format(response))
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory +
          '/{}_instructed_generation.png'.format(current_datetime),
          cv_array_bgr)
      self.save_data(
          self.log_directory +
          '/{}_instructed_generation.yaml'.format(current_datetime), {
              'instruction': prompt,
              'caption': '{}'.format(response)
          })
    if self.use_gui:
      cv2.imshow("LAVISServer Instructed Generation", cv_array_bgr)
      cv2.waitKey(1)
    return response

  def TextLocalization(self, request, context):
    model = self.models['text_localization']['model']
    device = self.models['text_localization']['device']
    vis_processors = self.models['text_localization']['vis_processors']
    text_processors = self.models['text_localization']['text_processors']
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv_array_rgb)
    img = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](self.translate_input_text(request.text))
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    whole_gradcam = gradcam[0][1]
    response = TextLocalizationResponse()
    response.heatmap.CopyFrom(cv_array_to_image_proto(
        np.float32(whole_gradcam)))
    logger.info("Get image with {} size with prompt {}".format(
        image_proto_to_cv_array(request.image).shape, request.text))
    norm_img = np.float64(raw_image) / 255
    gradcam = np.float64(gradcam[0][1])
    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)
    vis_image = cv2.cvtColor(np.uint8(avg_gradcam * 255), cv2.COLOR_RGB2BGR)
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory +
          '/{}_text_localization_input.png'.format(current_datetime),
          cv_array_bgr)
      self.save_data(
          self.log_directory +
          '/{}_text_localization_visualization.png'.format(current_datetime),
          vis_image)
      self.save_data(
          self.log_directory +
          '/{}_text_localization.yaml'.format(current_datetime),
          {'text': request.text})
    if self.use_gui:
      cv2.imshow("LAVISServer Text Localization", vis_image)
      cv2.waitKey(1)
    return response

  def VisualQuestionAnswering(self, request, context):
    # Retrieve models
    model = self.models['visual_question_answering']['model']
    device = self.models['visual_question_answering']['device']
    vis_processors = self.models['visual_question_answering']['vis_processors']
    text_processors = self.models['visual_question_answering'][
        'text_processors']
    # Input image
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv_array_rgb)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # Input question
    raw_question = request.question
    question = self.translate_input_text(raw_question)
    # Generate result
    result = model.predict_answers(samples={
        "image": image,
        "text_input": question
    },
                                   inference_method='generate')
    raw_answer = result[0]
    answer = self.translate_output_text(raw_answer) 
    response = VisualQuestionAnsweringResponse(
        answer=self.translate_output_text(answer))
    logger.info("Get image with {} size with question {} (translated to {})".format(
        image_proto_to_cv_array(request.image).shape, raw_question, question))
    logger.info("answer: {} translated to {}".format(raw_answer, answer))
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory + '/{}_vqa_input.png'.format(current_datetime),
          cv_array_bgr)
      cv2.putText(cv_array_bgr, f'question: {question}\nanswer: {raw_answer}', (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
      self.save_data(
          self.log_directory + '/{}_vqa.png'.format(current_datetime),
          cv_array_bgr)
      self.save_data(
          self.log_directory + '/{}_vqa_input.yaml'.format(current_datetime), {
              'raw_question': raw_question,
              'question': question,
              'raw_answer': raw_answer,
              'answer': answer,
          })
    if self.use_gui:
      cv2.imshow("LAVISServer Visual Question Answering", cv_array_bgr)
      cv2.waitKey(1)
    return response


def download_model():
  logging.basicConfig(level=logging.INFO)
  for key, value in model_list.items():
    name = value['model_name']
    model_type = value['model_type']
    load_model_and_preprocess(name=name,
                              model_type=model_type,
                              is_eval=True,
                              device='cpu')


def main_server():
  parser = argparse.ArgumentParser(description='LAVIS Server.')
  parser.add_argument('--port',
                      default=50051,
                      type=int,
                      help='Port for Grpc server')
  parser.add_argument('--use-gui',
                      action='store_true',
                      help='Show GUI Windows if set')
  parser.add_argument('--log-directory',
                      default=None,
                      help='Directory for log saving')
  parser.add_argument('--use-translator',
                      action='store_true',
                      help='If set, translator runs internally.')
  parser.add_argument('--device-image-captioning', default=None, type=int)
  parser.add_argument('--model-image-captioning', default=None, type=str)
  parser.add_argument('--device-instructed-generation', default=None, type=int)
  parser.add_argument('--model-instructed-generation', default=None, type=str)
  parser.add_argument('--device-text-localization', default=None, type=int)
  parser.add_argument('--model-text-localization', default=None, type=str)
  parser.add_argument('--device-visual-question-answering',
                      default=None,
                      type=int)
  parser.add_argument('--model-visual-question-answering',
                      default=None,
                      type=str)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)

  model_device_dict = {}
  if args.device_image_captioning is not None and args.model_image_captioning is not None:
    model_device_dict['image_captioning'] = {
        'device': 'cuda:{}'.format(args.device_image_captioning),
        'model_name': model_list[args.model_image_captioning]['model_name'],
        'model_type': model_list[args.model_image_captioning]['model_type']
    }
  if args.device_instructed_generation is not None and args.model_instructed_generation is not None:
    model_device_dict['instructed_generation'] = {
        'device':
            'cuda:{}'.format(args.device_instructed_generation),
        'model_name':
            model_list[args.model_instructed_generation]['model_name'],
        'model_type':
            model_list[args.model_instructed_generation]['model_type']
    }
  if args.device_text_localization is not None and args.model_text_localization is not None:
    model_device_dict['text_localization'] = {
        'device': 'cuda:{}'.format(args.device_text_localization),
        'model_name': model_list[args.model_text_localization]['model_name'],
        'model_type': model_list[args.model_text_localization]['model_type']
    }
  if args.device_visual_question_answering is not None and args.model_visual_question_answering is not None:
    model_device_dict['visual_question_answering'] = {
        'device':
            'cuda:{}'.format(args.device_visual_question_answering),
        'model_name':
            model_list[args.model_visual_question_answering]['model_name'],
        'model_type':
            model_list[args.model_visual_question_answering]['model_type']
    }

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  add_LAVISServerServicer_to_server(
      LAVISServer(use_gui=args.use_gui,
                  log_directory=args.log_directory,
                  model_device_dict=model_device_dict,
                  use_translator=args.use_translator), server)
  server.add_insecure_port('[::]:{}'.format(args.port))
  server.start()
  server.wait_for_termination()
