import logging

from LAVIS_grpc_server.lavis_server_pb2 import ImageCaptioningResponse
from LAVIS_grpc_server.lavis_server_pb2_grpc import LAVISServerServicer

logger = logging.getLogger(__name__)


class LAVISServer(LAVISServerServicer):

  def __init__(self):
    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    self.model, self.vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type="pretrain_opt2.7b",
        is_eval=True,
        device=self.device)
    logger.info('Initialized')

  def ImageCaptioning(self, request, context):
    logger.info('type of request: {}, request: {}'.format(
        type(request), request))
    logger.info('type of context: {}, context: {}'.format(
        type(context), context))

    response = ImageCaptioningResponse()
    return response
