"""Dataset loading utilities for ORL and Extended YaleB face datasets."""
import os

import numpy as np
from PIL import Image


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
    images, labels = [], []
    subject_dirs = [d for d in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, d))
                    and d not in exclude_dirs]
    for i, person in enumerate(subject_dirs):
        person_dir = os.path.join(root, person)
        for fname in sorted(os.listdir(person_dir)):
            if not fname.endswith('.pgm') or 'ambient' in fname.lower():
                continue
            img = Image.open(os.path.join(person_dir, fname)).convert('L')
            if reduce > 1:
                w, h = img.size
                img = img.resize((w // reduce, h // reduce))
            arr = np.asarray(img, dtype=np.float64).reshape((-1, 1))
            images.append(arr)
            labels.append(i)
    V = np.concatenate(images, axis=1)
    if normalize:
        V = V / 255.0
    return V, np.array(labels)


def load_orl(data_root, reduce=3):
    return load_dataset(os.path.join(data_root, 'ORL'), reduce=reduce)


def load_yaleb(data_root, reduce=4):
    # yaleB39 holds the 64 ambient-light captures (not a real subject)
    return load_dataset(os.path.join(data_root, 'CroppedYaleB'), reduce=reduce,
                        exclude_dirs={'yaleB39'})
