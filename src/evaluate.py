import torch
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_model(
    model, 
    data_loader, 
    device, 
    target_sr, 
    id_to_name=None, 
    save_predictions=False
):
    """
    Evaluates the model on a given dataloader.
    Calculates Accuracy, Macro F1, and optionally returns a DataFrame 
    of per-file predictions for error analysis.
    """
    model.eval()
    all_labels = []
    all_preds = []
    rows = []

    with torch.no_grad():
        for batch in data_loader:
            # Handle standard batches vs metadata-rich batches
            if len(batch) == 2:
                features, labels = batch
                audio_paths = [""] * labels.size(0)
                metadata_paths = [""] * labels.size(0)
                lang_ids = [-1] * labels.size(0)
                leading_silence_samples = [-1] * labels.size(0)
            else:
                (features, labels, audio_paths, metadata_paths, 
                 lang_ids, leading_silence_samples) = batch

            # Move to GPU/CPU
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

            # Log granular predictions if requested
            if save_predictions:
                for i in range(labels.size(0)):
                    true_label = int(labels[i].cpu().item())
                    pred_label = int(predicted[i].cpu().item())
                    
                    true_name = id_to_name.get(true_label, str(true_label)) if id_to_name else str(true_label)
                    pred_name = id_to_name.get(pred_label, str(pred_label)) if id_to_name else str(pred_label)
                    
                    silence_samples = int(leading_silence_samples[i])
                    silence_sec = silence_samples / target_sr if silence_samples >= 0 else None

                    rows.append({
                        "audio_path": audio_paths[i],
                        "metadata_path": metadata_paths[i],
                        "lang_id": int(lang_ids[i]),
                        "correct_label": true_label,
                        "predicted_label": pred_label,
                        "correct_label_name": true_name,
                        "predicted_label_name": pred_name,
                        "is_error": true_label != pred_label,
                        "prob_class_0": float(probs[i, 0].cpu().item()),
                        "prob_class_1": float(probs[i, 1].cpu().item()),
                        "leading_silence_samples": silence_samples,
                        "leading_silence_sec": silence_sec,
                    })

    # Calculate global metrics
    accuracy = 100.0 * sum(int(p == y) for p, y in zip(all_preds, all_labels)) / max(1, len(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    pred_df = pd.DataFrame(rows) if save_predictions else None
    
    return accuracy, macro_f1, pred_df