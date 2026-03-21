from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SingleVideoRecorder:
    path: str
    fps: int = 30

    def __post_init__(self) -> None:
        import imageio

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._writer = imageio.get_writer(
            self.path, fps=self.fps, macro_block_size=None
        )

    def add_frame(self, frame: np.ndarray) -> None:
        # enforce even dimensions for FFmpeg
        h, w = frame.shape[:2]
        if h % 2 == 1:
            frame = frame[:-1, :]
        if w % 2 == 1:
            frame = frame[:, :-1]
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
