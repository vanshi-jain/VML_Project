"""
This script is adapted from the smoothed-vit implementation:
Salman, H., Jain, S., Wong, E., & Madry, A. (2021). 
"Certified Patch Robustness via Smoothed Vision Transformers".
ArXiv preprint arXiv:2110.07719.

Modifications include:
- Integration with custom dataset (chairs)
- CV2 image saving utility
- Visualization via robustness `vis_tools`
- Adaptation of patch attacker and preprocessing modules

Author: Vanshika Jain
"""
# ---------------------- Imports ----------------------
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from copy import deepcopy

# Robustness & Custom Utilities
from robustness import datasets, defaults
from robustness.tools import vis_tools
from utils.transfer_utils import get_dataset_and_loaders, get_model, TRANSFER_DATASETS
from utils.attackerpaper import PatchAttacker as attacker
from utils.custom_models.preprocess import PreProcessor
from utils.smoothing import DerandomizedSmoother
from data.data import make_train_val

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(conflict_handler='resolve')
for args_group in [defaults.CONFIG_ARGS, defaults.MODEL_LOADER_ARGS, defaults.TRAINING_ARGS, defaults.PGD_ARGS]:
    parser = defaults.add_args_to_parser(args_group, parser)

# Custom arguments
parser.add_argument('--dataset', type=str)
parser.add_argument('--pytorch-pretrained', default=False, action='store_true')
parser.add_argument('--freeze-level', type=int, default=-1)
parser.add_argument('--drop-tokens', action='store_true', default=False)
parser.add_argument('--cifar-preprocess-type', type=str, default='none')
parser.add_argument('--model-path', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume-ckpt-name', type=str, default='checkpoint.pt.latest')
parser.add_argument('--subset', type=int, default=None)
parser.add_argument('--no-tqdm', type=int, default=0)
parser.add_argument('--no-replace-last-layer', action='store_true')
parser.add_argument('--additional-hidden', type=int, default=0)
parser.add_argument('--update-BN-stats', action='store_true')
parser.add_argument('--ablation-target', type=int, default=None)
parser.add_argument('--ablate-input', action='store_true', default=False)
parser.add_argument('--ablation-type', type=str, default='col')
parser.add_argument('--ablation-size', type=int, default=19)
parser.add_argument('--skip-store', action='store_true')
parser.add_argument('--certify', action='store_true')
parser.add_argument('--certify-out-dir', default='OUTDIR_CERT')
parser.add_argument('--certify-mode', default='both', choices=['both', 'row', 'col', 'block'])
parser.add_argument('--certify-ablation-size', type=int, default=19)
parser.add_argument('--certify-patch-size', type=int, default=16)
parser.add_argument('--certify-stride', type=int, default=1)
parser.add_argument('--batch-id', type=int, default=None)

# ---------------------- Utility Functions ----------------------
def save_image(path, img):
    """Save image using OpenCV with proper color and scale conversion."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    return cv2.imwrite(path, img)

def args_preprocess(args):
    """Preprocess arguments and ensure required values are set."""
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0
    if args.pytorch_pretrained:
        assert not args.model_path, 'Specify either pytorch_pretrained or model_path, not both.'
    
    ALL_DS = TRANSFER_DATASETS + ['imagenet', 'stylized_imagenet']
    assert args.dataset in ALL_DS
    assert args.exp_name is not None

    default_ds = args.dataset if args.dataset in datasets.DATASETS else "cifar"
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, datasets.DATASETS[default_ds])
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, datasets.DATASETS[default_ds])
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, datasets.DATASETS[default_ds])
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[default_ds])
    return args

# ---------------------- Main Logic ----------------------
def main(args):
    # Load Cobra dataset
    data_loader, _ = make_train_val(
        img_dir=r"./code_changes/1/data/cobradata/val",
        split=0.1, batch_size=args.batch_size, val_only=True
    )

    # Load dataset config and model
    ds, _, _ = get_dataset_and_loaders(args)
    model, _ = get_model(args, ds)
    model.eval()

    # Setup attacker
    attacker_model = attacker(
        deepcopy(model), [0., 0., 0.], [1., 1., 1.], {
            'epsilon': 1.0,
            'random_start': True,
            'steps': 150,
            'step_size': 0.05,
            'block_size': args.certify_ablation_size,
            'num_classes': ds.num_classes,
            'patch_l': args.certify_patch_size,
            'patch_w': args.certify_patch_size
        })

    # Attach preprocessor for input ablation and normalization
    model.normalizer = PreProcessor(
        normalizer=model.normalizer,
        ablation_size=args.certify_ablation_size, 
        upsample_type=args.cifar_preprocess_type,
        return_mask=args.drop_tokens,
        do_ablation=args.ablate_input,
        ablation_type=args.ablation_type,
        ablation_target=args.ablation_target,
    )

    model.eval()
    model = nn.DataParallel(model)

    # Setup smoother
    smoothed_model = DerandomizedSmoother(
        column_model=model,
        row_model=None,
        block_size=(args.certify_ablation_size, args.certify_ablation_size),
        stride=(args.certify_stride, args.certify_stride)
    )

    # Certification threshold
    m = args.certify_patch_size
    s = args.certify_ablation_size
    stride = args.certify_stride
    na = math.ceil((m + s - 1)/stride)
    gap = na * 2 + 1  # Factor 2

    # Evaluation metrics
    total_clean, total_smo, total_cert, n = 0, 0, 0, 0
    pbar = tqdm(data_loader)

    for i, (X, y) in enumerate(pbar):
        X, y = X.cuda(), y.cuda()
        clean_acc = (model(X)[0].max(1)[1] == y).float().sum().item()
        attacked = attacker_model.perturb(X, y, float('inf'), random_count=1).detach()

        y_pred, y_counts, _ = smoothed_model(attacked, return_mode="all", nclasses=ds.num_classes)
        smooth_acc = (y_pred == y).sum().item()

        # Certification check
        y_1st_vals, y_1st_idx = y_counts.kthvalue(ds.num_classes, dim=1)
        y_2nd_vals, y_2nd_idx = y_counts.kthvalue(ds.num_classes - 1, dim=1)
        certified = ((y == y_1st_idx) * (y_1st_vals >= y_2nd_vals + gap)).sum().item()

        # Visualize inputs and outputs (comment to disable)
        vis_tools.show_image_column([X.detach().cpu(), attacked.cpu()], [y.cpu(), y_pred.cpu()])

        # Update metrics
        n += X.size(0)
        total_clean += clean_acc
        total_smo += smooth_acc
        total_cert += certified

        pbar.set_description(f'Acc: {total_clean/n:.2f} | Smoo: {total_smo/n:.2f} | Cert: {total_cert/n:.2f}')

    # Final summary
    print("Total samples:", n)
    print(f"Clean Accuracy: {total_clean/n:.3f}")
    print(f"Smoothed Accuracy: {total_smo/n:.3f}")
    print(f"Certified Accuracy: {total_cert/n:.3f}")

# ---------------------- Entrypoint ----------------------
if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)
    main(args)