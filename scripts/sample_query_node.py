#!/usr/bin/env python

import logging

import rospy
from sensor_msgs.msg import Image
from sound_play.libsoundplay import SoundClient
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

from LAVIS_grpc_server.srv import (ImageCaptioning, ImageCaptioningRequest,
                                   VisualQuestionAnswering,
                                   VisualQuestionAnsweringRequest)


class SampleQueryNode:

  def __init__(self):

    rospy.wait_for_service('~image_captioning')
    self.msg_image = None
    self.sound_client = SoundClient()
    self.sub_image = rospy.Subscriber('~image', Image, self.callback_img)
    self.sub_speech = rospy.Subscriber('~speech_to_text',
                                       SpeechRecognitionCandidates,
                                       self.callback_speech)
    self.client_ic = rospy.ServiceProxy('~image_captioning', ImageCaptioning)
    self.client_vqa = rospy.ServiceProxy('~visual_question_answering',
                                         VisualQuestionAnswering)

    rospy.loginfo('Initialized.')

  def callback_img(self, msg):
    self.msg_image = msg

  def callback_speech(self, msg):
    req = VisualQuestionAnsweringRequest()
    req.image = self.msg_image
    req.question = msg.transcript[0]
    res = self.client_vqa(req)
    rospy.loginfo("A: {}".format(res.answer))
    self.sound_client.say(res.answer, blocking=True)


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('sample_query_node')
  node = SampleQueryNode()
  rospy.spin()
