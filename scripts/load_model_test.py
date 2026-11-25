from html import parser
import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import make_loader
from src.models import CNN1D3
from src.evaluate import evaluate_model


def build_eval_dataframe(full_df, config):
    task_name = config["experiment"]["active_task"]
    task_cfg = config["tasks"][task_name]

    if task_name == "binary":
        lang0_id, lang1_id, lang0_name, lang1_name = task_cfg["language_pairs"][0]

        filtered_df = full_df[full_df["lang_id"].isin([lang0_id, lang1_id])].copy()
        filtered_df["label"] = filtered_df["lang_id"].map({
            lang0_id: 0,
            lang1_id: 1
        }).astype(int)

        id_to_name = {
            0: lang0_name,
            1: lang1_name
        }

        num_classes = 2
        exp_name = f"{lang0_name}_{lang1_name}"

    elif task_name == "multiclass":
        selected_lang_ids = [lang[0] for lang in task_cfg["languages"]]

        label_mapping = {
            lang_id: idx
            for idx, lang_id in enumerate(selected_lang_ids)
        }

        id_to_name = {
            idx: lang_name
            for idx, (_, _, lang_name) in enumerate(task_cfg["languages"])
        }

        filtered_df = full_df[full_df["lang_id"].isin(selected_lang_ids)].copy()
        filtered_df["label"] = filtered_df["lang_id"].map(label_mapping).astype(int)

        num_classes = task_cfg["num_classes"]
        exp_name = "multiclass_6label"

    else:
        raise ValueError(f"Unknown task: {task_name}")

    test_df = filtered_df[filtered_df["new_split"] == "test"].copy()

    return test_df, id_to_name, num_classes, exp_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_6.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="results/lid_silence_experiment_outputs/models/best_multiclass_6label__trainDur_5.0s__trainTrim_0__devTrim_0.pth",
        help="Path to saved best model .pth"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Test duration in seconds. If not set, uses all config durations.test"
    )

    parser.add_argument(
        "--trim",
        action="store_true",
        help="Remove leading silence during evaluation"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output file name prefix"
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    target_sr = config["experiment"]["target_sr"]
    batch_size = config["experiment"]["batch_size"]
    num_workers = config["experiment"]["num_workers"]
    test_trim = args.trim

    base_path = PROJECT_ROOT / config["paths"]["base_data"]
    all_csv_paths = list(base_path.glob("*/*.csv"))

    if not all_csv_paths:
        raise FileNotFoundError(f"No CSV files found under {base_path}")

    full_df = pd.concat(
        [pd.read_csv(p) for p in all_csv_paths],
        ignore_index=True
    )

    test_df, id_to_name, num_classes, exp_name = build_eval_dataframe(
        full_df,
        config
    )

    model = CNN1D3(num_classes=num_classes).to(device)
    model_path = PROJECT_ROOT / args.model

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    print(f"Loaded model from: {model_path}")
    print(f"Task: {config['experiment']['active_task']}")
    print(f"Experiment: {exp_name}")
    print(f"Num classes: {num_classes}")
    print(f"Test samples: {len(test_df)}")
    print(f"Remove leading silence: {test_trim}")

    if args.duration is None:
        test_durations = config["durations"]["test"]
    else:
        test_durations = [args.duration]

    output_dir = PROJECT_ROOT / config["paths"]["output_dir"] / "saved_model_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for test_duration in test_durations:
        print("=" * 60)
        print(f"Testing duration: {test_duration}s")
        print("=" * 60)

        test_loader = make_loader(
            df=test_df,
            duration=test_duration,
            trim_leading=test_trim,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            target_sr=target_sr,
            return_metadata=True,
            silence_kwargs=config["audio"]
        )

        test_acc, test_f1, pred_df = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            target_sr=target_sr,
            id_to_name=id_to_name,
            save_predictions=True
        )

        print(f"Duration {test_duration}s | Test Accuracy: {test_acc:.2f}%")
        print(f"Duration {test_duration}s | Test Macro F1: {test_f1:.4f}")

        if args.output_name is None:
            trim_name = "trim1" if test_trim else "trim0"
            output_prefix = f"{exp_name}__testDur_{test_duration}s__{trim_name}"
        else:
            output_prefix = f"{args.output_name}__testDur_{test_duration}s"

        pred_path = output_dir / f"predictions_{output_prefix}.csv"
        error_path = output_dir / f"errors_{output_prefix}.csv"

        pred_df.to_csv(pred_path, index=False)
        pred_df[pred_df["is_error"]].to_csv(error_path, index=False)

        all_summaries.append({
            "task": config["experiment"]["active_task"],
            "experiment_name": exp_name,
            "model_path": str(model_path),
            "test_duration_sec": test_duration,
            "test_trim_leading": test_trim,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "num_classes": num_classes,
            "num_test_samples": len(test_df),
            "prediction_path": str(pred_path),
            "error_path": str(error_path)
        })

    trim_name = "trim1" if test_trim else "trim0"
    summary_path = output_dir / f"summary_{exp_name}__all_durations__{trim_name}.csv"

    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)

    print(f"Summary saved to: {summary_path}")


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--config",
#         type=str,
#         default="configs/config_6.yaml",
#         help="Path to config file"
#     )

#     parser.add_argument(
#         "--model",
#         type=str,
#         default="results/lid_silence_experiment_outputs/models/best_multiclass_6label__trainDur_5.0s__trainTrim_0__devTrim_0.pth",
#         help="Path to saved best model .pth"
#     )

#     parser.add_argument(
#         "--duration",
#         type=float,
#         default=None,
#         help="Test duration in seconds. If not set, uses config durations.test[0]"
#     )

#     parser.add_argument(
#         "--trim",
#         action="store_true",
#         help="Remove leading silence during evaluation"
#     )

#     parser.add_argument(
#         "--output-name",
#         type=str,
#         default=None,
#         help="Optional output file name prefix"
#     )

#     args = parser.parse_args()

#     config_path = PROJECT_ROOT / args.config

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running evaluation on: {device}")

#     target_sr = config["experiment"]["target_sr"]
#     batch_size = config["experiment"]["batch_size"]
#     num_workers = config["experiment"]["num_workers"]

#     if args.duration is None:
#         test_durations = config["durations"]["test"]
#     else:
#         test_durations = [args.duration]

#     for test_duration in test_durations:
#         print(f"Testing duration: {test_duration}s")

#         test_loader = make_loader(
#             df=test_df,
#             duration=test_duration,
#             trim_leading=test_trim,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             target_sr=target_sr,
#             return_metadata=True,
#             silence_kwargs=config["audio"]
#         )

#         test_acc, test_f1, pred_df = evaluate_model(
#             model=model,
#             data_loader=test_loader,
#             device=device,
#             target_sr=target_sr,
#             id_to_name=id_to_name,
#             save_predictions=True
#         )

#     print(f"Duration {test_duration}s | Acc: {test_acc:.2f}% | F1: {test_f1:.4f}")



#     test_trim = args.trim

#     base_path = PROJECT_ROOT / config["paths"]["base_data"]
#     all_csv_paths = list(base_path.glob("*/*.csv"))

#     if not all_csv_paths:
#         raise FileNotFoundError(f"No CSV files found under {base_path}")

#     full_df = pd.concat(
#         [pd.read_csv(p) for p in all_csv_paths],
#         ignore_index=True
#     )

#     test_df, id_to_name, num_classes, exp_name = build_eval_dataframe(
#         full_df,
#         config
#     )

#     print(f"Task: {config['experiment']['active_task']}")
#     print(f"Experiment: {exp_name}")
#     print(f"Num classes: {num_classes}")
#     print(f"Test samples: {len(test_df)}")
#     print(f"Test duration: {test_duration}s")
#     print(f"Remove leading silence: {test_trim}")

#     test_loader = make_loader(
#         df=test_df,
#         duration=test_duration,
#         trim_leading=test_trim,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         target_sr=target_sr,
#         return_metadata=True,
#         silence_kwargs=config["audio"]
#     )

#     model = CNN1D3(num_classes=num_classes).to(device)
#     model_path = PROJECT_ROOT / args.model

#     model.load_state_dict(
#         torch.load(model_path, map_location=device, weights_only=True)
#     )

#     print(f"Loaded model from: {model_path}")

#     test_acc, test_f1, pred_df = evaluate_model(
#         model=model,
#         data_loader=test_loader,
#         device=device,
#         target_sr=target_sr,
#         id_to_name=id_to_name,
#         save_predictions=True
#     )

#     print(f"Test Accuracy: {test_acc:.2f}%")
#     print(f"Test Macro F1: {test_f1:.4f}")

#     output_dir = PROJECT_ROOT / config["paths"]["output_dir"] / "saved_model_eval"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if args.output_name is None:
#         trim_name = "trim1" if test_trim else "trim0"
#         output_prefix = f"{exp_name}__testDur_{test_duration}s__{trim_name}"
#     else:
#         output_prefix = args.output_name

#     pred_path = output_dir / f"predictions_{output_prefix}.csv"
#     error_path = output_dir / f"errors_{output_prefix}.csv"
#     summary_path = output_dir / f"summary_{output_prefix}.csv"

#     pred_df.to_csv(pred_path, index=False)
#     pred_df[pred_df["is_error"]].to_csv(error_path, index=False)

#     summary = {
#         "task": config["experiment"]["active_task"],
#         "experiment_name": exp_name,
#         "model_path": str(model_path),
#         "test_duration_sec": test_duration,
#         "test_trim_leading": test_trim,
#         "test_accuracy": test_acc,
#         "test_macro_f1": test_f1,
#         "num_classes": num_classes,
#         "num_test_samples": len(test_df),
#         "prediction_path": str(pred_path),
#         "error_path": str(error_path)
#     }

#     pd.DataFrame([summary]).to_csv(summary_path, index=False)

#     print(f"Predictions saved to: {pred_path}")
#     print(f"Errors saved to: {error_path}")
#     print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()