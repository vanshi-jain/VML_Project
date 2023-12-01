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
from PIL import Image
from data import make_train_val

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

def save_image(path,img):
    img = cv2.cvtColor( img,cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    return cv2.imwrite(path, img)

def main(args):
    # my own data - cobra cropped chairs
    data_loader,_ = make_train_val(img_dir=r"C:/Users/vanshika/Desktop/VML Project/code_changes/1/data/cobradata/val",
                                             split=0.1,batch_size=args.batch_size,val_only=True)
    ds, _, _ = get_dataset_and_loaders(args)
    model, _ = get_model(args, ds)
    model.eval()
    # attacker model
    attacker_model = attacker(deepcopy(model), [0.,0.,0.],[1.,1.,1.], {
                'epsilon':1.0,
                'random_start':True,
                'steps':150,
                'step_size':0.05,
                'block_size':args.certify_ablation_size,
                'num_classes':ds.num_classes,
                'patch_l':args.certify_patch_size,
                'patch_w':args.certify_patch_size
            })    
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
    model = torch.nn.DataParallel(model)
    m = args.certify_patch_size
    s = args.certify_ablation_size
    stride = args.certify_stride
    smoothed_model = DerandomizedSmoother(
            column_model=model, 
            row_model=None, 
            block_size=(s,s), 
            stride=(stride,stride)
        )
  
    na = math.ceil((m + s - 1)/stride)
    factor = 2
    gap = na*factor + 1 
    
    total_clean = 0
    total_smo = 0
    total_cert = 0
    n = 0

    pbar = tqdm(data_loader)
    for i,(X,y) in enumerate(pbar): 
        # save_image(path='./smoothed-vit/raw/Label_'+ str(y[0].numpy()) + "_" + str(i)+ ".jpeg",
                #    img=X[0].permute(1,2,0).numpy())
        X,y = X.cuda(),y.cuda()
        # print(X.shape, y.shape )
        # print(model(X)[0].max(1)[1].shape)
        acc = (model(X)[0].max(1)[1] == y).float().mean().item()*X.size(0)
        # print("clean:", acc)
        attacked = attacker_model.perturb(X,y,float('inf'),random_count=1).detach()
        y_pred, y_counts, _ = smoothed_model(attacked, return_mode="all", nclasses=ds.num_classes)
        # print("pred:", y_pred)
        smooth_acc = (y_pred == y).sum().item()
        y_1st_vals, y_1st_idx = y_counts.kthvalue(ds.num_classes,dim=1)
        y_2nd_vals, y_2nd_idx = y_counts.kthvalue(ds.num_classes-1,dim=1)
        y_certified = (y == y_1st_idx)*(y_1st_vals >= y_2nd_vals + gap)
        certified = y_certified.detach().cpu().numpy().sum().item()
        vis_tools.show_image_column([X.detach().cpu(), attacked.cpu()], [y.cpu(), y_pred.cpu()])
        # save_image(path='./smoothed-vit/attacked/Label_'+ str(y_pred[0].cpu().numpy()) + "_" + str(i)+".jpeg",
                    # img = attacked[0].cpu().permute(1,2,0).numpy())        
        # print("clean acc:", acc, " smo:",smooth_acc, " certi:", certified)
        
        n += X.size(0)
        total_clean += acc
        total_cert += certified
        total_smo += smooth_acc

        pbar.set_description(f'Acc: {total_clean/n:.2f} Smoo: {total_smo/n:.2f} Cert: {total_cert/n:.2f}')

    print("total:",n)
    print(f"clean:{total_clean/n:.3f} smo:{total_smo/n:.3f} cert:{total_cert/n:.3f}")

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
