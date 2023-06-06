#!/usr/bin/env python

import logging

import grpc
import numpy as np
import rospy
from foundation_model_grpc_server.srv import (ImageCaptioning,
                                              ImageCaptioningResponse,
                                              VisualQuestionAnswering,
                                              VisualQuestionAnsweringResponse)

import foundation_model_grpc_interface.lavis_server_pb2 as lavis_server_pb2
import foundation_model_grpc_interface.lavis_server_pb2_grpc as lavis_server_pb2_grpc
from foundation_model_grpc_utils import cv_array_to_image_proto


class Node:

  def __init__(self):

    hostname = rospy.get_param('~grpc_hostname', 'localhost')
    port = rospy.get_param('~grpc_port', '50051')

    self.srv_ic = rospy.Service('~image_captioning', ImageCaptioning,
                                self.handler_ic)
    self.srv_vqa = rospy.Service('~visual_question_answering',
                                 VisualQuestionAnswering, self.handler_vqa)

    self.channel = grpc.insecure_channel('{}:{}'.format(hostname, port))
    self.stub = lavis_server_pb2_grpc.LAVISServerStub(self.channel)
    rospy.loginfo('Initialized.')

  def __del__(self):

    self.channel.close()

  def handler_ic(self, req):

    rospy.logdebug('image.encoding: {}'.format(req.image.encoding))
    cv_array_rgb = np.frombuffer(req.image.data, dtype=np.uint8).reshape(
        req.image.height, req.image.width, -1)
    grpc_request = lavis_server_pb2.ImageCaptioningRequest()
    grpc_request.image.CopyFrom(cv_array_to_image_proto(cv_array_rgb))
    result = self.stub.ImageCaptioning(grpc_request, wait_for_ready=True)
    response = ImageCaptioningResponse()
    response.caption = result.caption
    return response

  def handler_vqa(self, req):

    cv_array_rgb = np.frombuffer(req.image.data, dtype=np.uint8).reshape(
        req.image.height, req.image.width, -1)
    grpc_request = lavis_server_pb2.VisualQuestionAnsweringRequest(
        question=req.question)
    grpc_request.image.CopyFrom(cv_array_to_image_proto(cv_array_rgb))
    result = self.stub.VisualQuestionAnswering(grpc_request,
                                               wait_for_ready=True)
    response = VisualQuestionAnsweringResponse()
    response.answer = result.answer
    return response


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('lavis_grpc_bridge_server')
  node = Node()
  rospy.spin()
