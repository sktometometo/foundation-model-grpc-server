import logging

import cv2
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from LAVIS_grpc_server.lavis_server_pb2 import ImageCaptioningResponse
from LAVIS_grpc_server.lavis_server_pb2_grpc import LAVISServerServicer
from LAVIS_grpc_server.utils import image_proto_to_cv_array

logger = logging.getLogger(__name__)


class LAVISServer(LAVISServerServicer):

  def __init__(self, use_gui: bool):
    self.dst_size = (300, 300)
    self.use_gui = use_gui
    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    self.model, self.vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type="pretrain_opt2.7b",
        is_eval=True,
        device=self.device)
    logger.info('Initialized')

  def __del__(self):
    if self.use_gui:
      cv2.destroyAllWindows()

  def ImageCaptioning(self, request, context):
    cv_array_rgb = image_proto_to_cv_array(request.image)
    cv_array_bgr = cv2.cvtColor(cv_array_rgb, cv2.COLOR_RGB2BGR)
    if self.use_gui:
      cv2.imshow("LAVISServer", cv_array_bgr)
      cv2.waitKey(1)
    original_size = cv_array_rgb.shape[:2]
    raw_image = Image.fromarray(cv2.resize(cv_array_rgb, self.dst_size))
    image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
    result = self.model.generate({"image": image})
    response = ImageCaptioningResponse(caption=result[0])
    logger.info("Get image with {} size".format(cv_array_rgb.shape))
    logger.info("Generate caption: {}".format(response))
    return response
