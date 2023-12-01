from robustness import datasets, defaults
import torch
import torch.nn
from torchvision import transforms
import torch.nn.functional as F
import argparse
from tqdm.notebook import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
from robustness.tools import vis_tools
from utils.transfer_utils import get_dataset_and_loaders, get_model, TRANSFER_DATASETS
from utils.attackerpaper import PatchAttacker as attacker
from copy import deepcopy
import cv2
import glob
from utils.custom_models.preprocess import PreProcessor
from utils.smoothing import *
import cv2
from image_net_label_names import label_image

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str,)
parser.add_argument('--pytorch-pretrained', default=False, action='store_true')
parser.add_argument('--freeze-level', type=int, default=-1)
parser.add_argument('--drop-tokens', action='store_true', default=False)
parser.add_argument('--cifar-preprocess-type', type=str, default='none')
parser.add_argument('--model-path', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume-ckpt-name', type=str, default='checkpoint.pt.latest')
parser.add_argument('--subset', type=int, default=None)
parser.add_argument('--no-tqdm', type=int, default=0,)
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

def main(args):
    ds, _, _ = get_dataset_and_loaders(args)
    model, _ = get_model(args, ds)
    tcomp = transforms.Compose([
                transforms.ToTensor(),
                ])    
    # model.normalizer = PreProcessor(
    #     normalizer=model.normalizer,
    #     ablation_size=args.certify_ablation_size, 
    #     upsample_type=args.cifar_preprocess_type,
    #     return_mask=args.drop_tokens,
    #     do_ablation=args.ablate_input,
    #     ablation_type=args.ablation_type,
    #     ablation_target=args.ablation_target,
    # )
    model.eval() 
    model = torch.nn.DataParallel(model)
    m = args.certify_patch_size
    s = args.certify_ablation_size
    stride = args.certify_stride
    # smoothed_model = DerandomizedSmoother(
    #         column_model=model, 
    #         row_model=None, 
    #         block_size=(s,s), 
    #         stride=(stride,stride)
    #     )
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while cap.isOpened():
        # Capture frame-by-frame
        ret, original = cap.read()
        h,w,_ = original.shape
        frame = cv2.resize(original[100:400,100:400].copy(),(224,224))
        frame_tensor = tcomp(frame).unsqueeze(0)
        y = model(frame_tensor)[0].max(1)[1].detach().item()
        label = label_image[y]
        cv2.rectangle(original, (100, 100), (400, 400),(255,0,0), 2)
        cv2.putText(original,label,(h//2,w//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        # write the flipped frame
        out.write(original)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Display the resulting frameQ
        cv2.imshow('frame', original)
        if cv2.waitKey(1) == ord('q'):
            break    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()    

def args_preprocess(args):
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0
    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'
    ALL_DS = TRANSFER_DATASETS + ['imagenet', 'stylized_imagenet']
    assert args.dataset in ALL_DS
    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None
    # Preprocess args
    default_ds = args.dataset if args.dataset in datasets.DATASETS else "cifar"
    args =  defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, datasets.DATASETS[default_ds])
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, datasets.DATASETS[default_ds])
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, datasets.DATASETS[default_ds])
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[default_ds])
    return args

if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)
    main(args)
