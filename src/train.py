import time
import torch
from tqdm import tqdm
from pathlib import Path

# Import the evaluation helper to test the model at the end of each epoch
from src.evaluate import evaluate_model

def train_one_experiment(
    model, 
    train_loader, 
    dev_loaders, 
    criterion, 
    optimizer, 
    epochs, 
    device, 
    target_sr,
    logger, 
    run_name, 
    model_dir
):
    """
    Handles the standard PyTorch training loop for a single model configuration.
    Evaluates against multiple development sets (e.g., trimmed vs. untrimmed) 
    and saves the best model weights for each configuration.
    
    Returns:
        best_by_dev_trim (dict): Tracks the best accuracy, F1, and model path per dev set.
        train_time_sec (float): Total time taken to train the model.
    """
    # 1. Initialize tracking dictionary for model checkpoints
    model_dir = Path(model_dir)
    best_by_dev_trim = {
        dev_trim: {
            "best_dev_accuracy": -1.0,
            "best_dev_macro_f1": -1.0,
            "best_epoch": -1,
            "model_path": model_dir / f"best_{run_name}__devTrim_{int(dev_trim)}.pth",
        }
        for dev_trim in dev_loaders.keys()
    }

    train_start_time = time.time()

    # 2. Main Epoch Loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0

        for features, labels in tqdm(train_loader, desc=f"{run_name} | Epoch {epoch}/{epochs}", leave=False):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        epoch_time_sec = time.time() - epoch_start_time

        # --- VALIDATION PHASE ---
        dev_log_parts = []
        
        # Test against all provided dev set variations (e.g., silence trimmed vs untrimmed)
        for dev_trim, dev_loader in dev_loaders.items():
            
            # Note: save_predictions is False during training to speed up epochs
            dev_acc, dev_f1, _ = evaluate_model(
                model=model, 
                data_loader=dev_loader, 
                device=device, 
                target_sr=target_sr,
                save_predictions=False 
            )
            
            dev_log_parts.append(f"dev_trim={dev_trim}: acc={dev_acc:.2f}%, macro_f1={dev_f1:.4f}")

            # --- CHECKPOINT SAVING LOGIC ---
            best_info = best_by_dev_trim[dev_trim]
            
            # Update if accuracy is strictly better, OR if accuracy is tied but F1 is better
            if (dev_acc > best_info["best_dev_accuracy"]) or (
                dev_acc == best_info["best_dev_accuracy"] and dev_f1 > best_info["best_dev_macro_f1"]
            ):
                best_info["best_dev_accuracy"] = dev_acc
                best_info["best_dev_macro_f1"] = dev_f1
                best_info["best_epoch"] = epoch
                torch.save(model.state_dict(), best_info["model_path"])

        # 3. Log Epoch Results
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"epoch_time_sec={epoch_time_sec:.2f} | "
            + " | ".join(dev_log_parts)
        )

    # 4. Final Wrap-up
    train_time_sec = time.time() - train_start_time
    logger.info(f"Training finished: {run_name} | Total time: {train_time_sec:.2f}s")

    return best_by_dev_trim, train_time_sec