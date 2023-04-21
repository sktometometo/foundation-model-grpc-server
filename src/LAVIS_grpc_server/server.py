import logging

import cv2
import numpy as np
import torch
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from PIL import Image

from LAVIS_grpc_server.lavis_server_pb2 import (ImageCaptioningResponse,
                                                InstructedGenerationResponse,
                                                TextLocalizationResponse)
from LAVIS_grpc_server.lavis_server_pb2_grpc import LAVISServerServicer
from LAVIS_grpc_server.utils import (cv_array_to_image_proto,
                                     image_proto_to_cv_array)

logger = logging.getLogger(__name__)


class LAVISServer(LAVISServerServicer):

  def __init__(self,
               use_gui: bool,
               model_name="blip2_opt",
               model_type="pretrain_opt2.7b"):
    self.dst_size = (300, 300)
    self.use_gui = use_gui
    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=self.device)
    logger.info('Initialized')

  def __del__(self):
    if self.use_gui:
      cv2.destroyAllWindows()

  def ImageCaptioning(self, request, context):
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(cv_array_rgb, cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv2.resize(cv_array_rgb, self.dst_size))
    image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
    result = self.model.generate({"image": image})
    response = ImageCaptioningResponse(caption=result[0])
    logger.info("Get image with {} size".format(cv_array_rgb.shape))
    logger.info("Generate caption: {}".format(response))
    if self.use_gui:
      cv2.imshow("LAVISBLIP2Server Image Captioning", cv_array_bgr)
      cv2.waitKey(1)
    return response

  def InstructedGeneration(self, request, context):
    cv_array_rgb = image_proto_to_cv_array(request.image)
    prompt = request.prompt
    cv_array_bgr = cv2.cvtColor(cv_array_rgb, cv2.COLOR_RGB2BGR)
    raw_image = Image.fromarray(cv2.resize(cv_array_rgb, self.dst_size))
    image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
    result = self.model.generate({'image': image, 'prompt': prompt})
    response = InstructedGenerationResponse(response=result[0])
    logger.info("Get image with {} size with prompt {}".format(
        cv_array_rgb.shape, prompt))
    logger.info("Generate caption: {}".format(response))
    if self.use_gui:
      cv2.imshow("LAVISServer Instructed Generation", cv_array_bgr)
      cv2.waitKey(1)
    return response

  def TextLocalization(self, request, context):
    cv_array_rgb = image_proto_to_cv_array(request.image)
    raw_image = Image.fromarray(cv2.resize(cv_array_rgb, self.dst_size))
    img = self.vis_processors['eval'](raw_image)
    txt = self.text_processors["eval"](request.text)
    txt_tokens = self.model.tokenizer(txt, return_tensors="pt").to(self.device)
    gradcam, _ = compute_gradcam(self.model, img, txt, txt_tokens, block_num=7)
    whole_gradcam = gradcam[0][1]
    logger.info('whole_graphcam dtype: {}, shape: {}'.format(
        whole_gradcam.dtype, whole_gradcam.shape))
    response = TextLocalizationResponse()
    response.heatmap.CopyFrom(cv_array_to_image_proto(whole_gradcam))
    if self.use_gui:
      avg_gradcam = getAttMap(np.float32(raw_image), gradcam[0][1], blur=True)
      cv2.imshow("LAVISServer Text Localization",
                 cv2.cvtColor(avg_gradcam, cv2.COLOR_RGB2BGR))
      cv2.waitKey(1)
    return response
