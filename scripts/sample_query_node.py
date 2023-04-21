#!/usr/bin/env python

import logging

import rospy
from sensor_msgs.msg import Image
from sound_play.libsoundplay import SoundClient

from LAVIS_grpc_server.srv import ImageCaptioning, ImageCaptioningRequest


class SampleQueryNode:

  def __init__(self):

    rospy.wait_for_service('~image_captioning')
    self.sound_client = SoundClient()
    self.client = rospy.ServiceProxy('~image_captioning', ImageCaptioning)
    self.sub = rospy.Subscriber('~image', Image, self.callback)
    rospy.loginfo('Initialized.')

  def callback(self, msg):

    req = ImageCaptioningRequest()
    req.image = msg
    res = self.client(req)
    rospy.loginfo(res.caption)
    self.sound_client.say(res.caption, blocking=True)


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('sample_query_node')
  node = SampleQueryNode()
  rospy.spin()
