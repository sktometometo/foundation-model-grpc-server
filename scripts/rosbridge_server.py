#!/usr/bin/env python

import logging

import grpc

from cv_bridge import CvBridge

from LAVIS_grpc_server import lavis_server_pb2, lavis_server_pb2_grpc
from LAVIS_grpc_server.utils import cv_array_to_image_proto

import rospy
from LAVIS_grpc_server.srv import ImageCaptioning, ImageCaptioningResponse


class Node:

  def __init__(self):

    self.srv = rospy.Service('~image_captioning', ImageCaptioning, self.handler)
    self.channel = grpc.insecure_channel('localhost:50051')
    self.stub = lavis_server_pb2_grpc.LAVISServerStub(self.channel)
    self.cv_bridge = CvBridge()

  def __del__(self):

    self.channel.close()

  def handler(self, req):

    rospy.loginfo('image.encoding: {}'.format(req.image.encoding))

    grpc_request = lavis_server_pb2.ImageCaptioningRequest()
    grpc_request.image.CopyFrom(
        cv_array_to_image_proto(self.cv_bridge.imgmsg_to_cv2(req.image)))
    result = self.stub.ImageCaptioning(grpc_request)
    response = ImageCaptioningResponse()
    response.caption = result.caption
    return response


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('lavis_grpc_server')
  node = Node()
  rospy.spin()
