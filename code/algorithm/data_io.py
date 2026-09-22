"""Dataset loading utilities for the ORL and Extended YaleB face datasets.

The loaders return the data in the column-stacked convention used
throughout the assignment: ``V`` has one *column per image* and one row
per pixel, so that a factorisation ``V ~= WH`` reads as "``W`` holds the
parts-based basis images, ``H`` holds each image's coefficients".
"""
import os

import numpy as np
from PIL import Image

#: image geometry after the default downscaling, as (height, width)
ORL_SHAPE = (112, 92)
YALEB_SHAPE = (192, 168)


def resolve_data_root(data_root):
    """Accept either ``.../data`` or a folder that merely *contains* it.

    The repository has historically been run with ``--data-root ../data``
    and with ``--data-root ../data/data``, which silently produced
    FileNotFoundError for half the scripts.  Resolving the path once,
    here, removes that whole class of "it works on my machine" failure.
    """
    candidates = [
        data_root,
        os.path.join(data_root, 'data'),
    ]
    for cand in candidates:
        if (os.path.isdir(os.path.join(cand, 'ORL'))
                or os.path.isdir(os.path.join(cand, 'CroppedYaleB'))):
            return cand
    raise FileNotFoundError(
        f"Could not find 'ORL' or 'CroppedYaleB' under {data_root!r} "
        f"(also tried {os.path.join(data_root, 'data')!r}). "
        "Copy the two dataset folders into code/data/ or pass --data-root."
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
    if not images:
        raise FileNotFoundError(f'no .pgm images found under {root!r}')
    V = np.concatenate(images, axis=1)
    if normalize:
        V = V / 255.0
    return V, np.array(labels)


def load_orl(data_root, reduce=3):
    root = resolve_data_root(data_root)
    return load_dataset(os.path.join(root, 'ORL'), reduce=reduce)


def load_yaleb(data_root, reduce=4):
    """Load the Extended YaleB dataset: 2414 images of 38 subjects.

    The subject folders are yaleB01-yaleB13 and yaleB15-yaleB39 (there is
    no yaleB14), i.e. 38 folders.  Each folder holds 64 illumination
    captures plus exactly one ``*Ambient.pgm`` frame, which is the
    ambient-light reference rather than a face capture and is excluded by
    ``load_dataset``; 38 * 65 - 38 = 2414, matching the count quoted in
    the assignment brief.

    NOTE: an earlier version of this loader excluded the whole ``yaleB39``
    folder on the assumption that it held the ambient captures.  It does
    not -- it is a genuine 38th subject -- so that version silently
    trained on 2350 images of 37 classes.  Every number produced before
    this fix is therefore for a different dataset than the brief
    specifies and had to be recomputed.
    """
    root = resolve_data_root(data_root)
    return load_dataset(os.path.join(root, 'CroppedYaleB'), reduce=reduce)


def image_shape(dataset, reduce):
    """Post-downscale (height, width) for a dataset name."""
    base = {'orl': ORL_SHAPE, 'yaleb': YALEB_SHAPE}[dataset]
    return base[0] // reduce, base[1] // reduce


#: dataset name -> (loader, default reduce factor)
DATASETS = {
    'orl': (load_orl, 3),
    'yaleb': (load_yaleb, 4),
}


def load(dataset, data_root, reduce=None):
    """Load a dataset by name. Returns (V, Y, height, width)."""
    loader, default_reduce = DATASETS[dataset]
    reduce = default_reduce if reduce is None else reduce
    V, Y = loader(data_root, reduce=reduce)
    height, width = image_shape(dataset, reduce)
    if V.shape[0] != height * width:
        raise ValueError(
            f'{dataset}: expected {height * width} pixels per image, '
            f'got {V.shape[0]}')
    return V, Y, height, width
