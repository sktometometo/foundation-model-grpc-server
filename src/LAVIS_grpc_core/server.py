import datetime
import logging
import os

import cv2
import numpy as np
import torch
import yaml
from deep_translator import GoogleTranslator
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from PIL import Image

from LAVIS_grpc_interface.lavis_server_pb2 import (
    ImageCaptioningResponse, InstructedGenerationResponse,
    TextLocalizationResponse, VisualQuestionAnsweringResponse)
from LAVIS_grpc_interface.lavis_server_pb2_grpc import LAVISServerServicer
from LAVIS_grpc_utils import cv_array_to_image_proto, image_proto_to_cv_array

logger = logging.getLogger(__name__)


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
      return self.input_translator.translate(text)
    else:
      return text

  def translate_output_text(self, text):
    if self.use_translator:
      return self.output_translator.translate(text)
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
    model = self.models['image_captioning']['model']
    device = self.models['image_captioning']['device']
    vis_processors = self.models['image_captioning']['vis_processors']
    text_processors = self.models['image_captioning']['text_processors']
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv_array_rgb)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.generate({"image": image})
    response = ImageCaptioningResponse(
        caption=self.translate_output_text(result[0]))
    logger.info("Got image shape: {}".format(
        image_proto_to_cv_array(request.image).shape))
    logger.info("Generate caption: {}".format(response))
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory +
          '/{}_image_captioning.png'.format(current_datetime), cv_array_bgr)
      self.save_data(
          self.log_directory +
          '/{}_image_captioning.yaml'.format(current_datetime),
          {'caption': '{}'.format(response)})
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
    model = self.models['visual_question_answering']['model']
    device = self.models['visual_question_answering']['device']
    vis_processors = self.models['visual_question_answering']['vis_processors']
    text_processors = self.models['visual_question_answering'][
        'text_processors']
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image),
                                cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv_array_rgb)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.predict_answers(samples={
        "image": image,
        "text_input": self.translate_input_text(request.question)
    },
                                   inference_method='generate')
    response = VisualQuestionAnsweringResponse(
        answer=self.translate_output_text(result[0]))
    logger.info("Get image with {} size with prompt {}".format(
        image_proto_to_cv_array(request.image).shape, request.question))
    logger.info("answer: {}".format(result[0]))
    if self.log_directory is not None:
      current_datetime = datetime.datetime.now().isoformat()
      self.save_data(
          self.log_directory + '/{}_vqa.png'.format(current_datetime),
          cv_array_bgr)
      self.save_data(
          self.log_directory + '/{}_vqa.yaml'.format(current_datetime), {
              'question': request.question,
              'answer': result[0]
          })
    if self.use_gui:
      cv2.imshow("LAVISServer Visual Question Answering", cv_array_bgr)
      cv2.waitKey(1)
    return response
