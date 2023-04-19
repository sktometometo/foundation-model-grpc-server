import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

import cv2


class ContinuousServer:

    def __init__(self):

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=self.device
        )

    def inference(self, image):

        raw_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        result = self.model.generate({"image": image})
        return result


if __name__ == '__main__':

    p = ContinuousProcess()
    p.loop()
