"""Command-line entry point."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Train and run multi-view knee MRI models.")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser(
        "download",
        help="Download released classification and upstream preprocessing weights",
    )
    d.add_argument("--output", default="weights")
    for name in ["prepare", "predict"]:
        a = sub.add_parser(name)
        a.add_argument("--data", required=True)
        a.add_argument("--split", default="test")
        a.add_argument("--weights", default="weights")
        a.add_argument("--device", default="cuda:0")
        if name == "prepare":
            a.add_argument("--cache", required=True)
            a.add_argument("--size", type=int, choices=[320, 384], default=384)
            a.add_argument("--no-side-check", action="store_true")
        else:
            a.add_argument("--output", default="outputs/submission.csv")
            a.add_argument("--cache")
            a.add_argument("--chunk", type=int, default=64)
            a.add_argument(
                "--precision", choices=["fp16", "bf16", "fp32"], default="fp16"
            )
            a.add_argument(
                "--branch",
                choices=["ensemble", "llm", "distilled", "public"],
                default="ensemble",
            )
            a.add_argument("--studies", help="Optional JSON list of study identifiers")
    a = sub.add_parser("train")
    a.add_argument("--config", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--fold", type=int, required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--initial")
    a.add_argument("--teacher")
    a.add_argument("--teacher-weight", type=float, default=0.25)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--bias-labels")
    a = sub.add_parser(
        "teacher", help="Predict only the training partition for one student fold"
    )
    a.add_argument("--labels", required=True)
    a.add_argument("--fold", type=int, required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--chunk", type=int, default=64)
    a = sub.add_parser("evaluate")
    a.add_argument("--labels", required=True)
    a.add_argument("--predictions", required=True)
    a.add_argument("--output")
    a = sub.add_parser(
        "export", help="Export a selected training checkpoint as plain model tensors"
    )
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--output", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "download":
        from .download import download_weights

        download_weights(args["output"])
    elif command == "prepare":
        from .inference import prepare

        args["side_check"] = not args.pop("no_side_check")
        prepare(**args)
    elif command == "predict":
        from .inference import predict

        selection = args.pop("studies")
        args["study_ids"] = (
            json.loads(Path(selection).read_text()) if selection else None
        )
        predict(**args)
    elif command == "train":
        from .training import train

        args["config_path"] = args.pop("config")
        args["labels_path"] = args.pop("labels")
        train(**args)
    elif command == "teacher":
        from .training import read_labels
        from .inference import predict_cached, publish_predictions

        frame = read_labels(args["labels"])
        uids = frame.index[frame.fold != args["fold"]].tolist()
        values = predict_cached(
            args["checkpoint"],
            args["cache"],
            uids,
            device=args["device"],
            chunk=args["chunk"],
        )
        publish_predictions(args["output"], uids, values)
    elif command == "evaluate":
        import pandas as pd
        from .constants import UID, LABELS
        from .metrics import auc_metrics

        target = pd.read_csv(args["labels"], dtype={UID: str}).set_index(UID)
        pred = pd.read_csv(args["predictions"], dtype={UID: str}).set_index(UID)
        if not pred.index.is_unique or not pred.index.isin(target.index).all():
            raise ValueError("Unexpected prediction identifiers")
        report = auc_metrics(target.loc[pred.index, LABELS].values, pred[LABELS].values)
        report["cases"] = len(pred)
        text = json.dumps(report, indent=2)
        print(text)
        if args["output"]:
            Path(args["output"]).write_text(text + "\n")
    elif command == "export":
        import torch
        from .checkpoints import load_model, save_checkpoint

        model = load_model(args["checkpoint"], device="cpu")
        save_checkpoint(args["output"], model.state_dict())


if __name__ == "__main__":
    main()
