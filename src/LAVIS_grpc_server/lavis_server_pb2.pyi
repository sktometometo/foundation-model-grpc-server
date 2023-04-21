from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Image(_message.Message):
    __slots__ = ["encoding", "height", "image_data", "width"]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    encoding: int
    height: int
    image_data: bytes
    width: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., encoding: _Optional[int] = ..., image_data: _Optional[bytes] = ...) -> None: ...

class ImageCaptioningRequest(_message.Message):
    __slots__ = ["image"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: Image
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class ImageCaptioningResponse(_message.Message):
    __slots__ = ["caption"]
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    caption: str
    def __init__(self, caption: _Optional[str] = ...) -> None: ...

class InstructedGenerationRequest(_message.Message):
    __slots__ = ["image", "prompt"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    image: Image
    prompt: str
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ..., prompt: _Optional[str] = ...) -> None: ...

class InstructedGenerationResponse(_message.Message):
    __slots__ = ["response"]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class TextLocalizationRequest(_message.Message):
    __slots__ = ["image", "text"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    image: Image
    text: str
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ..., text: _Optional[str] = ...) -> None: ...

class TextLocalizationResponse(_message.Message):
    __slots__ = ["heatmap"]
    HEATMAP_FIELD_NUMBER: _ClassVar[int]
    heatmap: Image
    def __init__(self, heatmap: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class VisualQuestionAnsweringRequest(_message.Message):
    __slots__ = ["image", "question"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    image: Image
    question: str
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ..., question: _Optional[str] = ...) -> None: ...

class VisualQuestionAnsweringResponse(_message.Message):
    __slots__ = ["answer"]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    answer: str
    def __init__(self, answer: _Optional[str] = ...) -> None: ...
