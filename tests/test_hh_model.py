from transformers import pipeline
                                                                                                                                  

def test_forward():
    transcriber = pipeline(task="automatic-speech-recognition")
    transcript = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

