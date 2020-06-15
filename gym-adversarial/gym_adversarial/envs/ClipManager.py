from shutil import rmtree
from tempfile import mkdtemp
from pathlib import Path
from contextlib import contextmanager
import os
from matplotlib import pyplot as plt


def filename_generator(d: Path):
    i = 0
    while True:
        yield d / f"filename{str(i).zfill(3)}.png"
        i += 1


class PlotSaver:
    def __init__(self, d: Path):
        self.i = 0
        self.d = d

    def add_img(self, img):
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.title(f"step {self.i}")
        self.i += 1
        plt.savefig(self.d / f"filename{str(self.i).zfill(3)}.png")


@contextmanager
def clip_manager(clip_file: Path):
    d = Path(mkdtemp(suffix=None, prefix="CLIP_", dir=None))
    clip_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        clip_file.unlink()
    except FileNotFoundError:
        pass
    try:
        yield PlotSaver(d)
    finally:
        s = f"{d}/filename%3d.png"
        os.system(f"ffmpeg -framerate 4 -i {s} -r 30 -pix_fmt yuv420p {clip_file}")
        # rmtree(d, ignore_errors=True)
