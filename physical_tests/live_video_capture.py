import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from robustness import datasets, defaults
from utils.transfer_utils import get_dataset_and_loaders, get_model, TRANSFER_DATASETS
from data.image_net_labels import label_image

# ───────────────────────────────────────────────────────────── #
# Argument Setup
# ───────────────────────────────────────────────────────────── #
def build_arg_parser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

    # Custom args
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pytorch-pretrained', default=False, action='store_true')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--certify-patch-size', type=int, default=16)
    parser.add_argument('--certify-ablation-size', type=int, default=19)
    parser.add_argument('--certify-stride', type=int, default=1)
    
    return parser

# ───────────────────────────────────────────────────────────── #
# Argument Preprocessing
# ───────────────────────────────────────────────────────────── #
def preprocess_args(args):
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0
    if args.pytorch_pretrained:
        assert not args.model_path, 'Specify only one: pytorch_pretrained OR model_path.'

    assert args.dataset in TRANSFER_DATASETS + ['imagenet', 'stylized_imagenet']
    assert args.exp_name is not None

    default_ds = args.dataset if args.dataset in datasets.DATASETS else "cifar"
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, datasets.DATASETS[default_ds])
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, datasets.DATASETS[default_ds])
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, datasets.DATASETS[default_ds])
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[default_ds])
    return args

# ───────────────────────────────────────────────────────────── #
# Model and Dataset Loading
# ───────────────────────────────────────────────────────────── #
def load_model_and_dataset(args):
    ds, _, _ = get_dataset_and_loaders(args)
    model, _ = get_model(args, ds)
    model.eval()
    return torch.nn.DataParallel(model)

# ───────────────────────────────────────────────────────────── #
# Video Capture Logic
# ───────────────────────────────────────────────────────────── #
def capture_and_classify(model, transform):
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (640, 480))

    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or failed to read frame.")
                break

            prediction, display_frame = process_frame(frame, model, transform)
            out.write(display_frame)

            cv2.imshow('Real-time Prediction', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# ───────────────────────────────────────────────────────────── #
# Frame Processing
# ───────────────────────────────────────────────────────────── #
def process_frame(frame, model, transform):
    h, w, _ = frame.shape
    cropped = cv2.resize(frame[100:400, 100:400], (224, 224))
    tensor_input = transform(cropped).unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor_input)[0].argmax(1).item()
        label = label_image[prediction]

    # Annotate
    annotated = frame.copy()
    cv2.rectangle(annotated, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.putText(annotated, label, (h // 2, w // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    return label, annotated

# ───────────────────────────────────────────────────────────── #
# Entry Point
# ───────────────────────────────────────────────────────────── #
def main():
    parser = build_arg_parser()
    args = preprocess_args(parser.parse_args())
    model = load_model_and_dataset(args)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    capture_and_classify(model, transform)

if __name__ == "__main__":
    main()
