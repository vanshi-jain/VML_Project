# VML_Project
Code to Reality: A Study on Certified Patch Robustness

Train:
```
python main.py --dataset cifar10 --data /tmp --arch deit_tiny_patch16_224 --pytorch-pretrained --out-dir OUTDIR2 --exp-name vitT --epochs 30 --lr 0.01 --step-lr 10 --batch-size 128 --weight-decay 5e-4 --adv-train 0 --freeze-level -1 --drop-tokens --cifar-preprocess-type simple224 --ablate-input --ablation-type col --ablation-size 4
```
Certify:
```
python main.py --dataset cifar10 --data /tmp --arch deit_tiny_patch16_224 --out-dir OUTDIR2 --exp-name vitT --batch-size 128 --adv-train 0 --freeze-level -1 --drop-tokens --cifar-preprocess-type simple224 --resume --eval-only 1 --certify --certify-out-dir OUTDIR_CERT --certify-mode col --certify-ablation-size 4 --certify-patch-size 4 --resume-ckpt-name 29_checkpoint.pt
```

Use run_attack.py for custom dataset
```
python run_attack.py --dataset imagenet --data ./code_changes/1/data/imagnet --arch deit_tiny_patch16_224 --out-dir smoothed-vit --exp-name Imagenet-models --batch-size 1 --adv-train 0 --freeze-level -1  --resume --eval-only 1 --certify-ablation-size 19 --certify-patch-size 16 --resume-ckpt-name deit_tiny_k19.pt
```
