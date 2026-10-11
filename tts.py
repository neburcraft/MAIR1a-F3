"""
This file has been created with AI assistance as it is not part of
the core assignment and the sherpa-onnx documentation is confusing
- Ruben
"""

import os
import numpy as np
import sherpa_onnx
import sounddevice as sd
import urllib.request
import tarfile


class TTS:
  """Wrapper for offline sherpa_onnx text-to-speech"""
  tts: sherpa_onnx.OfflineTts
  stream: sd.OutputStream

  def __init__(self):
    model_dir = "vits-piper-en_US-amy-low"

    # Download small offline TTS model if not already present and clean up afterwards
    if not os.path.exists(model_dir):
      urllib.request.urlretrieve(
          f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{model_dir}.tar.bz2",
          f"{model_dir}.tar.bz2")
      tar = tarfile.open(f"{model_dir}.tar.bz2")
      tar.extractall(filter='data')
      tar.close()
      os.remove(f"{model_dir}.tar.bz2")

    # Configure the offline TTS Engine settings
    config = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, "en_US-amy-low.onnx"),
            lexicon=os.path.join(model_dir, "lexicon.txt"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            data_dir=os.path.join(model_dir, "espeak-ng-data"),
        ),
        num_threads=2,
        debug=False,
    ))

    self.tts = sherpa_onnx.OfflineTts(config)

  def play_audio_chunk(self, samples: np.ndarray) -> int:
    """
    This function gets audio chunks from sherpa-onnx during generation
    and streams them directly to the sounddevice
    """
    # Write raw floating-point audio data straight to your speakers
    self.stream.write(samples.astype(np.float32))
    return 1 # Continue generating

  def say(self, text: str):
    """Convert text to speech using sherpa_onnx and play the audio"""
    sample_rate = self.tts.sample_rate
    self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
    self.stream.start()

    # Callback plays audio as soon as it is generated
    self.tts.generate(text, callback=self.play_audio_chunk)

    self.stream.stop()
    self.stream.close()
