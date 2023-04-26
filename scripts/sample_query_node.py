#!/usr/bin/env python

import logging

import cv2
import numpy as np
import rospy
from LAVIS_grpc_server.srv import (ImageCaptioning, ImageCaptioningRequest,
                                   VisualQuestionAnswering,
                                   VisualQuestionAnsweringRequest)
from sensor_msgs.msg import Image
from sound_play.libsoundplay import SoundClient
from speech_recognition_msgs.msg import SpeechRecognitionCandidates


class SampleQueryNode:

  def __init__(self):
    self.sound_client = SoundClient()

    self.camera_type = rospy.get_param(
        '~camera_type', 'subscriber')  # options, 'subscriber', 'opencv'

    if self.camera_type == 'subscriber':
      self.msg_image = None
      self.sub_image = rospy.Subscriber('~image', Image, self.callback_img)
    elif self.camera_type == 'opencv':
      video_device_index = int(rospy.get_param('~video_device_index', 0))
      self.cam_cv = cv2.VideoCapture(video_device_index)
    else:
      rospy.logerr('Unknown camera_type: {}'.format(self.camera_type))
      raise ValueError('Please use valid camera_type')

    self.task_type = rospy.get_param(
        '~task_type', 'vqa')  # options: 'vqa', 'image_captioning'

    if self.task_type == 'image_captioning':
      rospy.wait_for_service('~image_captioning')
      self.client_ic = rospy.ServiceProxy('~image_captioning', ImageCaptioning)
    elif self.task_type == 'vqa':
      rospy.wait_for_service('~visual_question_answering')
      self.client_vqa = rospy.ServiceProxy('~visual_question_answering',
                                           VisualQuestionAnswering)
    else:
      rospy.logerr('Unknown task_type: {}'.format(self.task_type))
      raise ValueError('Please use valid task_type')

    self.sub_speech = rospy.Subscriber('~speech_to_text',
                                       SpeechRecognitionCandidates,
                                       self.callback_speech)

    rospy.loginfo('Initialized.')

  def callback_img(self, msg):
    self.msg_image = msg

  def callback_speech(self, msg):

    if self.camera_type == 'subscriber':
      image = self.msg_image
    elif self.camera_type == 'opencv':
      ret, frame = self.cam_cv.read()
      cv_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image = Image()
      image.height = cv_array.shape[0]
      image.width = cv_array.shape[1]
      image.encoding = '8UC3'
      image.data.frombytes(cv_array.tobytes())
      image.step = len(image.data) // image.header

    req = VisualQuestionAnsweringRequest()
    req.image = image
    req.question = msg.transcript[0]
    rospy.loginfo("Q: {}".format(req.question))
    res = self.client_vqa(req)
    rospy.loginfo("A: {}".format(res.answer))
    self.sound_client.say(res.answer, blocking=True)


if __name__ == '__main__':

  logging.basicConfig(level=logging.INFO)
  rospy.init_node('sample_query_node')
  node = SampleQueryNode()
  rospy.spin()
