import argparse
import datetime
import logging
import os
from concurrent import futures

import cv2
import deep_translator
import grpc
import llama
import numpy as np
import yaml
from deep_translator import GoogleTranslator
from foundation_model_grpc_interface.lavis_server_pb2 import (
    ImageCaptioningResponse,
    VisualQuestionAnsweringResponse,
)
from foundation_model_grpc_interface.lavis_server_pb2_grpc import (
    LAVISServerServicer,
    add_LAVISServerServicer_to_server,
)
from foundation_model_grpc_utils import image_proto_to_cv_array
from llama.llama_adapter import _MODELS
from llama.utils import _download
from PIL import Image

logger = logging.getLogger(__name__)


class LLaMAAdapterServer(LAVISServerServicer):
    def __init__(
        self,
        llama_dir,
        log_directory=None,
        use_translator=False,
        target_language="ja",
        checkpoint_cache="$HOME/.cache/llama_adapter_pretrained/",
    ):
        self.log_directory = log_directory
        self.use_translator = use_translator
        self.input_translator = GoogleTranslator(source="auto", target="en")
        self.output_translator = GoogleTranslator(source="auto", target=target_language)

        self.device = "cuda"
        os.makedirs(os.path.expandvars(checkpoint_cache), exist_ok=True)
        self.model, self.preprocess = llama.load(
            "BIAS-7B", llama_dir, self.device, download_root=os.path.expandvars(checkpoint_cache)
        )

        logger.info("Initialized")

    def translate_input_text(self, text):
        if self.use_translator:
            try:
                logger.info("original input text: {}".format(text))
                return self.input_translator.translate(text)
            except deep_translator.exceptions.NotValidPayload as error:
                logger.error("Error: {}".format(error))
                return text
        else:
            return text

    def translate_output_text(self, text):
        if self.use_translator:
            try:
                logger.info("original output text: {}".format(text))
                return self.output_translator.translate(text)
            except deep_translator.exceptions.NotValidPayload as error:
                logger.error("Error: {}".format(error))
                return text
        else:
            return text

    def save_data(self, path: str, data):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if type(data) == np.ndarray:
            cv2.imwrite(path, data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, encoding="utf-8", allow_unicode=True)

    def ImageCaptioning(self, request, context):
        # Input Image
        cv_array_rgb = image_proto_to_cv_array(request.image)
        cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image), cv2.COLOR_RGB2BGR)
        raw_image = Image.fromarray(cv_array_rgb)
        image = self.preprocess(raw_image).unsqueeze(0).to(self.device)
        # Input question
        prompt = llama.format_prompt("Describe the image in detail.")
        logger.info(f"Get request image (shape {image.shape})")
        logger.info(f"prompt: {prompt}")
        # Generate inference
        result = self.model.generate(image, [prompt])
        raw_caption = result[0]
        caption = self.translate_output_text(raw_caption)
        logger.info(f'Got answer "{raw_caption}" (translated to "{caption}")')
        if self.log_directory is not None:
            current_datetime = datetime.datetime.now().isoformat()
            self.save_data(
                os.path.join(
                    self.log_directory, "{}_image_captioning_input.png".format(current_datetime)
                ),
                cv_array_bgr,
            )
            cv2.putText(
                cv_array_bgr,
                f"caption: {raw_caption}",
                (0, 0),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self.save_data(
                os.path.join(
                    self.log_directory, "{}_image_captioning.png".format(current_datetime)
                ),
                cv_array_bgr,
            )
            self.save_data(
                os.path.join(
                    self.log_directory, "{}_image_captioning_input.yaml".format(current_datetime)
                ),
                {
                    "raw_caption": raw_caption,
                    "caption": caption,
                },
            )
        response = ImageCaptioningResponse(caption=caption)
        return response

    def VisualQuestionAnswering(self, request, context):
        # Input Image
        cv_array_rgb = image_proto_to_cv_array(request.image)
        cv_array_bgr = cv2.cvtColor(image_proto_to_cv_array(request.image), cv2.COLOR_RGB2BGR)
        raw_image = Image.fromarray(cv_array_rgb)
        image = self.preprocess(raw_image).unsqueeze(0).to(self.device)
        # Input question
        raw_question = request.question
        question = self.translate_input_text(raw_question)
        prompt = llama.format_prompt(question)
        # logging
        logger.info(
            f'Get request image (shape {image.shape}) with question "{raw_question}" (translated to "{question}")'
        )
        logger.info(f"prompt: {prompt}")
        # Generate inference
        result = self.model.generate(image, [prompt])
        raw_answer = result[0]
        answer = self.translate_output_text(raw_answer)
        logger.info(f'Got answer "{raw_answer}" (translated to "{answer}")')
        if self.log_directory is not None:
            current_datetime = datetime.datetime.now().isoformat()
            self.save_data(
                os.path.join(self.log_directory, "{}_vqa_input.png".format(current_datetime)),
                cv_array_bgr,
            )
            cv2.putText(
                cv_array_bgr,
                f"question: {question}\nanswer: {raw_answer}",
                (0, 0),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self.save_data(
                os.path.join(self.log_directory, "{}_vqa.png".format(current_datetime)),
                cv_array_bgr,
            )
            self.save_data(
                os.path.join(self.log_directory, "{}_vqa_input.yaml".format(current_datetime)),
                {
                    "raw_question": raw_question,
                    "question": question,
                    "prompt": prompt,
                    "raw_answer": raw_answer,
                    "answer": answer,
                },
            )
        response = VisualQuestionAnsweringResponse(answer=answer)
        return response


def download_model(name="BIAS-7B", checkpoint_cache="$HOME/.cache/llama_adapter_pretrained/"):
    directory = os.path.expandvars("$HOME/.cache/llama_adapter_pretrained/")
    os.makedirs(directory, exist_ok=True)
    _download(_MODELS[name], directory)


def run_server():
    parser = argparse.ArgumentParser(description="LLaMA Adapter Server.")
    parser.add_argument("--port", default=50051, type=int, help="Port for Grpc server")
    parser.add_argument("--log-directory", default=None, help="Directory for log saving")
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument(
        "--use-translator", action="store_true", help="If set, translator runs internally."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_LAVISServerServicer_to_server(
        LLaMAAdapterServer(
            llama_dir=args.llama_dir,
            log_directory=args.log_directory,
            use_translator=args.use_translator,
        ),
        server,
    )
    server.add_insecure_port("[::]:{}".format(args.port))
    server.start()
    server.wait_for_termination()
