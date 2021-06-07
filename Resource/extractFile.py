import os
os.getcwd()
import tarfile

from pathlib import Path
extracted_to_path = Path.cwd() / 'sampleFolder'

with tarfile.open('Models/centernet_hourglass_512x512_1.tar.gz') as tar:
    tar.extractall(path='C://Uni/EC/VideoTracker/CENTERNET_MODELS')