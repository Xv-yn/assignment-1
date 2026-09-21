"""Dataset loading utilities for ORL and Extended YaleB face datasets."""
import os
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_data_root(data_root):
    """Return the directory that directly contains ORL and CroppedYaleB.

    Some local copies unpack the datasets as ``data/ORL`` and
    ``data/CroppedYaleB``; this repository currently has them under
    ``data/data``. Accepting both layouts keeps every experiment script usable
    with either the assignment grader's layout or the checked-out repo layout.
    """
    root = Path(data_root)
    candidates = [root, root / "data"]
    for candidate in candidates:
        if (candidate / "ORL").is_dir() and (candidate / "CroppedYaleB").is_dir():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find ORL and CroppedYaleB under "
        f"{root!s} or {root / 'data'!s}."
    )


def load_dataset(root, reduce=3, normalize=True, exclude_dirs=()):
    """Load a face image dataset into a column-stacked matrix.

    Args:
        root:   dataset folder; each subject is a subfolder of .pgm images.
        reduce: integer downscale factor applied to every image.
        normalize: if True, scale pixel values from [0, 255] to [0, 1].
        exclude_dirs: subject folder names to skip (keeps labels dense 0..K-1).

    Returns:
        V: float64 array (n_pixels, n_images), data matrix.
        Y: int array (n_images,), class label per image (subject index).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    images, labels = [], []
    subject_dirs = [d for d in sorted(os.listdir(root))
                    if os.path.isdir(root / d)
                    and d not in exclude_dirs]
    for i, person in enumerate(subject_dirs):
        person_dir = root / person
        for fname in sorted(os.listdir(person_dir)):
            if not fname.endswith('.pgm') or 'ambient' in fname.lower():
                continue
            img = Image.open(person_dir / fname).convert('L')
            if reduce > 1:
                w, h = img.size
                img = img.resize((w // reduce, h // reduce))
            arr = np.asarray(img, dtype=np.float64).reshape((-1, 1))
            images.append(arr)
            labels.append(i)
    if not images:
        raise ValueError(f"No .pgm face images found under {root}")
    V = np.concatenate(images, axis=1)
    if normalize:
        V = V / 255.0
    return V, np.array(labels)


def load_orl(data_root, reduce=3):
    return load_dataset(Path(resolve_data_root(data_root)) / 'ORL', reduce=reduce)


def load_yaleb(data_root, reduce=4):
    # Extended YaleB has 38 subjects: yaleB01-13 and yaleB15-39.
    # Ambient captures are already skipped by load_dataset via the filename check.
    return load_dataset(Path(resolve_data_root(data_root)) / 'CroppedYaleB',
                        reduce=reduce)
