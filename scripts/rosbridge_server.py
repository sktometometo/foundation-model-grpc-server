#!/usr/bin/env python

import logging

import grpc
import numpy as np
import rospy

from LAVIS_grpc_server import lavis_server_pb2, lavis_server_pb2_grpc
from LAVIS_grpc_server.srv import ImageCaptioning, ImageCaptioningResponse
from LAVIS_grpc_server.utils import cv_array_to_image_proto


class Node:

  def __init__(self):

    self.srv = rospy.Service('~image_captioning', ImageCaptioning, self.handler)
    self.channel = grpc.insecure_channel('localhost:50051')
    self.stub = lavis_server_pb2_grpc.LAVISServerStub(self.channel)
    rospy.loginfo('Initialized.')

  def __del__(self):

    self.channel.close()

  def handler(self, req):

    rospy.loginfo('image.encoding: {}'.format(req.image.encoding))
    cv_array_rgb = np.frombuffer(req.image.data, dtype=np.uint8).reshape(
        req.image.height, req.image.width, -1)
    grpc_request = lavis_server_pb2.ImageCaptioningRequest()
    grpc_request.image.CopyFrom(cv_array_to_image_proto(cv_array_rgb))
    result = self.stub.ImageCaptioning(grpc_request)
    response = ImageCaptioningResponse()
    response.caption = result.caption
    return response


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('lavis_grpc_bridge_server')
  node = Node()
  rospy.spin()
