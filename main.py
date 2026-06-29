import argparse

from src.dataset.dataset import run_preprocess
from src.training.train import run_train
from src.evaluation.evaluate import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["preprocess", "train", "eval"])
    parser.add_argument("--model", default="faster_rcnn")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    if args.mode == "preprocess":
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_preprocess(cfg)
    elif args.mode == "train":
        run_train(args.model, args.config)
    elif args.mode == "eval":
        run_eval(args.model, args.config)


if __name__ == "__main__":
    main()