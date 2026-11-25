import sys
import yaml
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import make_loader
from src.models import CNN1D3
from src.evaluate import evaluate_model

# 1. Global Language Mapping (based on your previous setup)
LANG_NAME_TO_ID = {
    "Mandarin": 0,
    "Cantonese": 1,
    "Japanese": 2,
    "German": 3,
    "Dutch": 4,
    "English": 5
}

if __name__ == "__main__":
    # 2. Setup Config and Device
    config_path = PROJECT_ROOT / "configs" / "config_eval_fleurs.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Batch Evaluation on: {device}")

    # 3. Load the NEW Dataset
    base_path = PROJECT_ROOT / config["paths"]["base_data"]
    all_csv_paths = list(base_path.glob("*/*.csv"))
    
    if not all_csv_paths:
        raise FileNotFoundError(f"No CSV files found in {base_path}")

    full_df = pd.concat([pd.read_csv(p) for p in all_csv_paths], ignore_index=True)
    
    # Optional: Only evaluate on the "test" split of the new dataset
    full_test_df = full_df[full_df["new_split"] == "test"].copy()
    print(f"Loaded {len(full_test_df)} test samples from new dataset.")

    # 4. Locate Models
    model_dir = PROJECT_ROOT / "results" / "lid_silence_experiment_outputs" / "models"
    model_paths = list(model_dir.glob("*.pth"))
    print(f"Found {len(model_paths)} models to evaluate.\n")

    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []

    # 5. Master Evaluation Loop
    for model_path in model_paths:
        filename = model_path.stem # e.g., best_Mandarin_Cantonese__trainDur_5.0s__devTrim_0
        
        # Parse filename to get the language pair
        parts = filename.split("__")
        lang_part = parts[0].replace("best_", "") # e.g., Mandarin_Cantonese
        lang_names = lang_part.split("_")
        
        # Safety check: ensure it's a binary model we can parse
        if len(lang_names) != 2 or lang_names[0] not in LANG_NAME_TO_ID:
            print(f"Skipping {filename} (Could not parse language pair)")
            continue

        lang0_name, lang1_name = lang_names
        lang0_id = LANG_NAME_TO_ID[lang0_name]
        lang1_id = LANG_NAME_TO_ID[lang1_name]

        print(f"{'='*60}")
        print(f"Evaluating Model: {filename}")
        print(f"Task: {lang0_name} vs {lang1_name}")

        # Filter the new dataset for this specific model's task
        pair_df = full_test_df[full_test_df["lang_id"].isin([lang0_id, lang1_id])].copy()
        pair_df["label"] = pair_df["lang_id"].map({lang0_id: 0, lang1_id: 1}).astype(int)
        
        id_to_name = {0: lang0_name, 1: lang1_name}

        # Initialize and load model weights
        model = CNN1D3(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Evaluate across requested durations and trim options
        for test_duration in config["durations"]["test"]:
            for test_trim in config["trim_options"]["test"]:
                
                test_loader = make_loader(
                    df=pair_df,
                    duration=test_duration,
                    trim_leading=test_trim,
                    batch_size=config["experiment"]["batch_size"],
                    shuffle=False,
                    num_workers=config["experiment"]["num_workers"],
                    target_sr=config["experiment"]["target_sr"],
                    return_metadata=False,
                    silence_kwargs=config["audio"]
                )

                test_acc, test_f1, _ = evaluate_model(
                    model=model,
                    data_loader=test_loader,
                    device=device,
                    target_sr=config["experiment"]["target_sr"],
                    id_to_name=id_to_name,
                    save_predictions=False # Set to True if you want granular CSVs for every single run
                )

                print(f" -> Dur: {test_duration}s | Trim: {test_trim} | Acc: {test_acc:.2f}% | F1: {test_f1:.4f}")

                all_summaries.append({
                    "model_name": filename,
                    "lang_pair": lang_part,
                    "test_duration_sec": test_duration,
                    "test_trim_leading": test_trim,
                    "test_accuracy": test_acc,
                    "test_macro_f1": test_f1,
                })

        # Continuously save summary so you don't lose data if it crashes
        pd.DataFrame(all_summaries).to_csv(output_dir / "batch_evaluation_summary.csv", index=False)

        # Clear the GPU memory
        del model
        torch.cuda.empty_cache()

    print("\n🎉 Batch evaluation complete! Check the new_dataset_evaluations folder.")