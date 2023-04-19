#!/usr/bin/env python
import cv2
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class ContinuousProcess:

  def __init__(self):

    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    self.model, self.vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type="pretrain_opt2.7b",
        is_eval=True,
        device=self.device)
    self.cap = cv2.VideoCapture(0)

  def __del__(self):

    self.cap.release()
    cv2.destroyAllWindows()

  def loop(self):

    while True:
      ret, frame = self.cap.read()
      raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(
          self.device)
      #result = self.model.generate({"image": image})
      result = self.model.generate({
          "image": image,
          "prompt": "Question: how many people are there? Answer:"
      })
      print(result)
      cv2.imshow("test", frame)
      if cv2.waitKey(1) != -1:
        break


if __name__ == '__main__':

  p = ContinuousProcess()
  p.loop()
