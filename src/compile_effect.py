import time
from abc import ABC, abstractmethod

import torch
from datasets import load_dataset
from tabulate import tabulate
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

NUMBER_OF_RUN = 1


class Model(ABC):
    @abstractmethod
    def run(): ...

    def compile(self):
        return torch.compile(self._model)


class T5(Model):
    def __init__(self, compile=False):
        super().__init__()
        self._model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        if compile:
            self._model = self.compile()

        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self._task = tokenizer(
            "translate English to German: The house is wonderful.", return_tensors="pt"
        ).input_ids

    def run(self):
        self._model.generate(self._task)


class Whisper(Model):
    def __init__(self, compile=False):
        super().__init__()
        # load model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self._model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        )
        self._model.config.forced_decoder_ids = None

        if compile:
            self._model = self.compile()

        # load dummy dataset and read audio files
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        self._task = processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features

    def run(self):
        self._model.generate(self._task)


class DETR(Model):
    def __init__(self, compile=False):
        super().__init__()
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self._model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        if compile:
            self._model = self.compile()
        self._task = image_processor(images=image, return_tensors="pt")

    def run(self):
        self._model(**self._task)


def run(model):
    start_time = time.time()
    model.run()
    end_time = time.time()

    duration = end_time - start_time

    return duration


def duration_test(number_of_run: int, model: Model, compile: bool):
    results = []
    compiled_flag = "1" if compile else "0"
    for iteration in range(number_of_run + 1):  # 最初の1回目はウォームアップ
        duration = run(model)
        results.append(
            [model.__class__.__name__, iteration + 1, compiled_flag, duration]
        )

    return results


def main():
    models = [T5, Whisper, DETR]
    runs = []

    for model in models:
        model_not_compiled = model(compile=False)
        not_compiled_test = duration_test(
            NUMBER_OF_RUN, model=model_not_compiled, compile=False
        )
        runs += not_compiled_test

        model_compiled = model(compile=True)
        compiled_tests = duration_test(
            NUMBER_OF_RUN, model=model_compiled, compile=True
        )
        runs += compiled_tests

    print(
        tabulate(
            runs,
            headers=["Model", "Iteration", "compiled", "Duration[sec]"],
            floatfmt=".5f",
        )
    )


if __name__ == "__main__":
    main()
