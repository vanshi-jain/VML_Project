# From Code to Reality: A Study on Certified Robustness Against Adversarial Attacks using CNNs and Vision Transformers

This repository contains the code and experiments for the study on **certified robustness against adversarial patches**, focusing on both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). The project explores robustness across the entire ML lifecycle, from training to real-world testing.

Check out my [blog](https://sites.google.com/view/jain-van-profile/projects/certified-ml) for a deeper understanding.

## 📌 Abstract

This study primarily explores the robustness within the realm of **adversarial patches**, encompassing diverse modifications confined to a small, contiguous area. While ensuring model robustness with formal guarantees can be intriguing, the more captivating aspect lies in **verifying models throughout their entire lifecycle**, from training to deployment.

I certified CNN and ViT models on CIFAR-10, confirming improved guarantees. Subsequent adversarial patch tests aligned closely with reported results. Extending to real-world datasets, I explored patch robustness with a focus on **chair images**, yielding unexpected insights and paving the way for practical applications. In a **physical-world test**, printed images subjected to patches revealed less favorable outcomes, but the experiment highlighted important areas for future exploration.

## 🧪 Experiments Summary

| S.No. | Description                                                       | Takeaways                           |
|-------|-------------------------------------------------------------------|-----------------------------------|
| 1     | Train Models with Certification for ResNet-18 and ViT-T           | Improved Certified Accuracy       |
| 2     | Attack and Certify Robustness on CIFAR-10 and ImageNet           | Nearly matched reported results   |
| 3     | Patch Robustness on Real-World Dataset (Chairs)                  | Surprising Results                |
| 4     | Real-World Patch Attack on Printed Posters                       | Interesting Physical-World Behavior |

## 🔍 Key Contributions

- Evaluated CNN and ViT robust models on CIFAR-10 using randomized smoothing techniques.
- Extended robustness testing to real-world image datasets involving **chair** objects.
- Performed physical-world robustness validation using **printed adversarial patch posters**.
- Achieved **up to 80% certified accuracy** on chair datasets using ViT models with patch size 16.
- Optimized ViT architecture for **faster inference** while maintaining standard accuracy.

## 📊 Results

**Certification on Real-World Dataset (Intel RealSense Camera, fine-tuned on chairs):**

| Model                         | Images | Inference Time | Patch Size | Std. Accuracy | Smoothed | Certified Accuracy |
|------------------------------|--------|----------------|------------|----------------|----------|---------------------|
| ViT-T + C100                 | 70     | -              | 2          | -              | -        | -                   |
| ViT-T + C100 + Chair Dataset | 500    | 1.7 sec        | 16         | 78.0           | 97.0     | 80.0                |
| ViT-T + C100 + Chair Dataset | 500    | 1.7 sec        | 32         | 79.0           | 97.0     | 46.0                |

## 🏁 Conclusion

This work demonstrates a **substantial enhancement** in certified robustness against adversarial patches by integrating **Visual Transformers (ViTs)** into the smoothing framework. The proposed approach maintains accuracy on clean images while significantly boosting robustness and reducing inference time, making ViTs a **practical alternative** to standard models in safety-critical applications.

Further, this study validates these techniques in **real-world and physical-world settings**, highlighting both the promise and the challenges of robust ML models.

## 🚀 How to Run

To install all dependencies:

```
pip install -r requirements.txt
```

To train a certified model on CIFAR-10:

```
python main.py --dataset cifar10 --data /tmp --arch deit_tiny_patch16_224 --pytorch-pretrained --out-dir OUTDIR2 --exp-name vitT --epochs 30 --lr 0.01 --step-lr 10 --batch-size 128 --weight-decay 5e-4 --adv-train 0 --freeze-level -1 --drop-tokens --cifar-preprocess-type simple224 --ablate-input --ablation-type col --ablation-size 4
```

To evaluate adversarial patch robustness:

```
python main.py --dataset cifar10 --data /tmp --arch deit_tiny_patch16_224 --out-dir OUTDIR2 --exp-name vitT --batch-size 128 --adv-train 0 --freeze-level -1 --drop-tokens --cifar-preprocess-type simple224 --resume --eval-only 1 --certify --certify-out-dir OUTDIR_CERT --certify-mode col --certify-ablation-size 4 --certify-patch-size 4 --resume-ckpt-name 29_checkpoint.pt
```

To run on your custom dataset:

```
python run_attack.py --dataset imagenet --data ./code_changes/1/data/imagnet --arch deit_tiny_patch16_224 --out-dir smoothed-vit --exp-name Imagenet-models --batch-size 1 --adv-train 0 --freeze-level -1  --resume --eval-only 1 --certify-ablation-size 19 --certify-patch-size 16 --resume-ckpt-name deit_tiny_k19.pt
```

## 📸 Real-World Testing

Printed poster experiments were performed by generating patched images, printing them, and capturing re-scanned versions via an Intel RealSense camera. See physical_tests/ for setup instructions.

## 📚 References

- [Randomized Smoothing for Certified Robustness](https://arxiv.org/abs/1902.02918)

- [Vision Transformers](https://arxiv.org/abs/2010.11929)

- [Certified Defenses against Adversarial Patches](https://arxiv.org/abs/2003.06693)
