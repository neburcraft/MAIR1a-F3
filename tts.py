"""
This file has been generated with heavy AI assistance as it is not part
of the core assignment and the sherpa-onnx documentation is not very
understandable. - Ruben
"""

import os
import numpy as np
import sherpa_onnx
import sounddevice as sd
import urllib.request
import tarfile

class TTS:
  tts: sherpa_onnx.OfflineTts
  stream: sd.OutputStream

  def __init__(self):
    # 1. Define where your downloaded model directory is located
    model_dir = "vits-piper-en_US-amy-low"

    # Download small offline TTS model if not already present
    if not os.path.exists(model_dir):
      urllib.request.urlretrieve(
          f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{model_dir}.tar.bz2",
          f"{model_dir}.tar.bz2"
      )
      tar = tarfile.open(f"{model_dir}.tar.bz2")
      tar.extractall(filter='data')
      tar.close()
      os.remove(f"{model_dir}.tar.bz2")

    # 2. Configure the Offline TTS Engine settings
    config = sherpa_onnx.OfflineTtsConfig(
      model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
          model=os.path.join(model_dir, "en_US-amy-low.onnx"),
          lexicon=os.path.join(model_dir, "lexicon.txt"),
          tokens=os.path.join(model_dir, "tokens.txt"),
          data_dir=os.path.join(model_dir, "espeak-ng-data"),
        ),
        num_threads=2,
        debug=False,
      )
    )

    # Initialize the engine
    self.tts = sherpa_onnx.OfflineTts(config)

  # 4. Define the callback function that executes as text gets synthesized
  def play_audio_chunk(self, samples: np.ndarray, progress: float) -> int:
    """
    This function intercepts audio chunks from sherpa-onnx mid-generation
    and streams them straight to the sounddevice hardware buffer.
    """
    # Write raw floating-point audio data straight to your speakers
    self.stream.write(samples.astype(np.float32))
    
    # Return 1 to tell sherpa-onnx to keep generating, 0 would cancel it
    return 1

  def say(self, text: str):
    # 3. Create a real-time playback stream using sounddevice
    # Note: sherpa-onnx neural models generate audio at a sample rate of 22050Hz
    sample_rate = self.tts.sample_rate

    self.stream = sd.OutputStream(
      samplerate=sample_rate, 
      channels=1, 
      dtype='float32'
    )
    self.stream.start()

    # Generating with a progress_callback makes it functional for live streaming
    self.tts.generate(text, callback=self.play_audio_chunk)

    # Gracefully close the hardware stream when done
    self.stream.stop()
    self.stream.close()

if __name__=="__main__":
  tts = TTS()
  tts.say("This is a test sentence.")