"""NMF algorithm implementations (numpy only)."""
from .data_io import load_dataset, load_orl, load_yaleb
from .evaluate import clustering_metrics, relative_reconstruction_error
from .noise import occlusion_noise
from .nmf_mu import nmf_frobenius
from .nmf_l1 import nmf_l1
from .nmf_l21 import nmf_l21

# name -> algorithm callable; k = n_components
ALGORITHMS = {
    'F-norm MU': nmf_frobenius,
    'L1-NMF': nmf_l1,
    'L2,1-NMF': nmf_l21,
}
