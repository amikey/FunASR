"""Initialize funasr package."""

import os

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()

from funasr.bin.inference_cli import infer