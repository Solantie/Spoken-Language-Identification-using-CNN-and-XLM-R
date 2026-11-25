import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path

# Add the project root to the Python path so it can find the 'src' module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    setup_logger, 
    compute_and_save_global_silence_stats, 
    get_pair_avg_ssr
)
from src.dataset import make_loader
from src.models import CNN1D3
from src.train import train_one_experiment
from src.evaluate import evaluate_model

if __name__ == "__main__":
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    task_name = config["experiment"]["active_task"]
    task_cfg = config["tasks"][task_name]

    NUM_CLASSES = task_cfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    logger = setup_logger(output_dir)
    logger.info(f"🚀 Starting Robustness Pipeline. Running on: {device}")

    # Extract hyperparameters
    TARGET_SR = config["experiment"]["target_sr"]
    BATCH_SIZE = config["experiment"]["batch_size"]
    EPOCHS = config["experiment"]["epochs"]
    LR = config["experiment"]["learning_rate"]
    NUM_WORKERS = config["experiment"]["num_workers"]

    # ==========================================
    # 2. DATA LOADING & GLOBAL STATS
    # ==========================================
    base_path = PROJECT_ROOT / config["paths"]["base_data"]
    all_csv_paths = list(base_path.glob("*/*.csv"))
    
    if not all_csv_paths:
        logger.error(f"No CSV files found at {base_path}. Check your config!")
        sys.exit(1)

    full_df = pd.concat([pd.read_csv(p) for p in all_csv_paths], ignore_index=True)
    logger.info(f"Loaded {len(full_df)} rows from {len(all_csv_paths)} CSV files.")

    # Compute global stats (using silence kwargs from config)
    global_stats_df = None
    
    if config["experiment"].get("compute_silence_stats", True):
        global_stats_df = compute_and_save_global_silence_stats(
            full_df, 
            target_sr=TARGET_SR, 
            output_dir=output_dir, 
            logger=logger, 
            **config["audio"]
        )
    else:
        logger.info("Skipping global silence statistics computation (flag set to false).")

    experiment_rows = []

    # ==========================================
    # 3. MASTER LANGUAGE PAIR LOOP
    # ==========================================
    experiment_rows = []


    if task_name == "binary":
        experiments = []

        for pair in task_cfg["language_pairs"]:
            lang0_id, lang1_id, lang0_name, lang1_name = pair

            filtered_df = full_df[full_df["lang_id"].isin([lang0_id, lang1_id])].copy()
            filtered_df["label"] = filtered_df["lang_id"].map({
                lang0_id: 0,
                lang1_id: 1
            }).astype(int)

            experiments.append({
                "name": f"{lang0_name}_{lang1_name}",
                "df": filtered_df,
                "id_to_name": {0: lang0_name, 1: lang1_name}
            })

    elif task_name == "multiclass":
        experiments = []
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

        experiments = [{
            "name": "multiclass_6label",
            "df": filtered_df,
            "id_to_name": id_to_name
        }]

    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    for exp in experiments:
        exp_name = exp["name"]
        filtered_df = exp["df"]
        id_to_name = exp["id_to_name"]

        train_df = filtered_df[filtered_df["new_split"] == "train"].copy()
        dev_df = filtered_df[filtered_df["new_split"] == "dev"].copy()
        test_df = filtered_df[filtered_df["new_split"] == "test"].copy()

        avg_ssr_pair = get_pair_avg_ssr(global_stats_df, filtered_df)

        logger.info("=" * 70)
        logger.info(f"🔥 STARTING EXPERIMENT: {exp_name}")
        logger.info(f"Task: {task_name} | Num classes: {NUM_CLASSES}")
        logger.info(f"Data -> Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
        logger.info(f"Average SSR: {avg_ssr_pair}")
        logger.info("=" * 70)

    
    # for pair in config["language_pairs"]:
    #     lang0_id, lang1_id, lang0_name, lang1_name = pair
    #     exp_name = f"{lang0_name}_{lang1_name}"
    #     id_to_binary_name = {0: lang0_name, 1: lang1_name}

    #     # Filter dataset for this specific pair
    #     filtered_df = full_df[full_df["lang_id"].isin([lang0_id, lang1_id])].copy()
    #     filtered_df["binary_label"] = filtered_df["lang_id"].map({lang0_id: 0, lang1_id: 1}).astype(int)

    #     train_df = filtered_df[filtered_df["new_split"] == "train"].copy()
    #     dev_df = filtered_df[filtered_df["new_split"] == "dev"].copy()
    #     test_df = filtered_df[filtered_df["new_split"] == "test"].copy()

    #     avg_ssr_pair = get_pair_avg_ssr(global_stats_df, filtered_df)

    #     logger.info("=" * 70)
    #     logger.info(f"🔥 STARTING EXPERIMENT: {exp_name}")
    #     logger.info(f"Data -> Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    #     logger.info(f"Average SSR: {avg_ssr_pair}")
    #     logger.info("=" * 70)

        # Iterate over Training Durations and Trim Options
        for train_duration in config["durations"]["train"]:
            for train_trim in config["trim_options"]["train"]:
                run_name = f"{exp_name}__trainDur_{train_duration}s__trainTrim_{int(train_trim)}"
                
                # Setup Loaders
                train_loader = make_loader(
                    df=train_df, duration=train_duration, trim_leading=train_trim, 
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
                    target_sr=TARGET_SR, silence_kwargs=config["audio"]
                )

                dev_loaders = {
                    dev_trim: make_loader(
                        df=dev_df, duration=train_duration, trim_leading=dev_trim, 
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                        target_sr=TARGET_SR, silence_kwargs=config["audio"]
                    )
                    for dev_trim in config["trim_options"]["dev"]
                }

                # Initialize Model, Criterion, Optimizer
                model = CNN1D3(num_classes=NUM_CLASSES).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LR)

                # Train the model
                best_by_dev_trim, train_time_sec = train_one_experiment(
                    model=model, train_loader=train_loader, dev_loaders=dev_loaders,
                    criterion=criterion, optimizer=optimizer, epochs=EPOCHS, 
                    device=device, target_sr=TARGET_SR, logger=logger, 
                    run_name=run_name, model_dir=(output_dir / "models")
                )

                # ==========================================
                # 4. TESTING PHASE
                # ==========================================
                for dev_trim, best_info in best_by_dev_trim.items():
                    test_model = CNN1D3(num_classes=NUM_CLASSES).to(device)
                    test_model.load_state_dict(torch.load(best_info["model_path"], map_location=device, weights_only=True))
                    
                    for test_duration in config["durations"]["test"]:
                        for test_trim in config["trim_options"]["test"]:
                            
                            test_loader = make_loader(
                                df=test_df, duration=test_duration, trim_leading=test_trim, 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                                target_sr=TARGET_SR, return_metadata=True, silence_kwargs=config["audio"]
                            )

                            test_acc, test_f1, pred_df = evaluate_model(
                                model=test_model, data_loader=test_loader, device=device, 
                                target_sr=TARGET_SR, id_to_name=id_to_name, save_predictions=True
                            )

                            # Save granular predictions
                            pred_base = f"{run_name}__testDur_{test_duration}s__devTrim_{int(dev_trim)}__testTrim_{int(test_trim)}"
                            all_pred_path = output_dir / "predictions" / f"predictions_{pred_base}.csv"
                            err_pred_path = output_dir / "predictions" / f"errors_{pred_base}.csv"
    
                            pred_df.to_csv(all_pred_path, index=False)
                            pred_df[pred_df["is_error"]].to_csv(err_pred_path, index=False)

                            # Append to master summary
                            experiment_rows.append({
                                "task": task_name,
                                "exp_name": exp_name,
                                "train_duration_sec": train_duration,
                                "test_duration_sec": test_duration,
                                "train_trim_leading": train_trim,
                                "dev_trim_leading": dev_trim,
                                "test_trim_leading": test_trim,
                                "best_epoch": best_info["best_epoch"],
                                "training_time_sec": train_time_sec,
                                "best_dev_accuracy": best_info["best_dev_accuracy"],
                                "best_dev_macro_f1": best_info["best_dev_macro_f1"],
                                "test_accuracy": test_acc,
                                "test_macro_f1": test_f1,
                                "average_ssr": avg_ssr_pair,
                                "best_model_path": str(best_info["model_path"])
                            })

                            logger.info(
                                f"RESULT | pair={exp_name} | train_dur={train_duration}s | test_dur={test_duration}s | "
                                f"trim(T/D/Te)={int(train_trim)}/{int(dev_trim)}/{int(test_trim)} | "
                                f"dev_acc={best_info['best_dev_accuracy']:.2f}% | test_acc={test_acc:.2f}% | test_f1={test_f1:.4f}"
                            )

                            # Update summary file continuously in case of a crash
                            pd.DataFrame(experiment_rows).to_csv(output_dir / "experiment_summary.csv", index=False)

    logger.info("🎉 All experiments finished flawlessly.")