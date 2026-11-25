import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. LOGGING SETUP
# ==========================================
def setup_logger(output_dir):
    """
    Sets up the logger and automatically creates the required subdirectories
    so the main execution script doesn't crash when saving files.
    """
    out_path = Path(output_dir)
    
    # Create the directory structure automatically
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "models").mkdir(parents=True, exist_ok=True)
    (out_path / "predictions").mkdir(parents=True, exist_ok=True)
    (out_path / "silence_stats").mkdir(parents=True, exist_ok=True)

    log_path = out_path / "experiment_results.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True, # Overrides any existing loggers
    )
    return logging.getLogger(__name__)


# ==========================================
# 2. AUDIO PROCESSING
# ==========================================
def load_audio_mono_resample(audio_path, target_sr):
    """Loads audio, enforces mono channel, and resamples to target_sr."""
    waveform, sr = torchaudio.load(str(audio_path))

    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
        sr = target_sr

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sr


def crop_or_pad(waveform, max_samples):
    """Ensures waveform is exactly max_samples long."""
    if waveform.shape[1] > max_samples:
        return waveform[:, :max_samples]
    pad_amount = max_samples - waveform.shape[1]
    return F.pad(waveform, (0, pad_amount))


# ==========================================
# 3. SILENCE DETECTION MATH
# ==========================================
def compute_leading_silence_samples(
    waveform,
    sample_rate,
    db_threshold=-40.0,
    frame_ms=20.0,
    hop_ms=10.0,
    eps=1e-10,
):
    """
    Calculates how many samples of silence exist at the start of the waveform.
    Uses RMS energy in dBFS compared against the db_threshold.
    """
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {waveform.shape}")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    num_samples = waveform.shape[1]
    if num_samples == 0:
        return 0

    frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * hop_ms / 1000.0))

    # If too short for frame-based detection, classify whole audio based on RMS.
    if num_samples < frame_length:
        rms = torch.sqrt(torch.mean(waveform ** 2) + eps).item()
        db = 20.0 * torch.log10(torch.tensor(rms + eps)).item()
        return 0 if db > db_threshold else num_samples

    frames = waveform.unfold(dimension=1, size=frame_length, step=hop_length)  # [1, n_frames, frame]
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + eps).squeeze(0)          # [n_frames]
    db = 20.0 * torch.log10(rms + eps)

    speech_frames = torch.nonzero(db > db_threshold, as_tuple=False)
    if speech_frames.numel() == 0:
        return num_samples

    first_speech_frame = int(speech_frames[0].item())
    leading_samples = first_speech_frame * hop_length
    return min(leading_samples, num_samples)


def trim_leading_silence(waveform, sample_rate, **silence_kwargs):
    """Helper to physically cut the silence off the tensor."""
    leading_samples = compute_leading_silence_samples(waveform, sample_rate, **silence_kwargs)
    return waveform[:, leading_samples:], leading_samples


# ==========================================
# 4. STATISTICS & AGGREGATION
# ==========================================
def silence_stats_for_row(row, target_sr, **silence_kwargs):
    """Calculates all silence-to-speech ratios for a single CSV row."""
    audio_path = row["audio_path"]
    waveform, sr = load_audio_mono_resample(audio_path, target_sr=target_sr)

    original_samples = int(waveform.shape[1])
    leading_silence_samples = int(compute_leading_silence_samples(waveform, sample_rate=sr, **silence_kwargs))
    speech_samples_after_leading = max(0, original_samples - leading_silence_samples)

    leading_silence_sec = leading_silence_samples / sr
    speech_duration_sec = speech_samples_after_leading / sr
    original_duration_sec = original_samples / sr

    if speech_samples_after_leading > 0:
        silence_to_speech_ratio = leading_silence_samples / speech_samples_after_leading
    else:
        silence_to_speech_ratio = float("inf")

    result = {
        "audio_path": audio_path,
        "metadata_path": row.get("metadata_path", ""),
        "new_split": row.get("new_split", ""),
        "lang_id": row.get("lang_id", ""),
        "original_samples": original_samples,
        "original_duration_sec": original_duration_sec,
        "leading_silence_samples": leading_silence_samples,
        "leading_silence_sec": leading_silence_sec,
        "speech_samples_after_leading": speech_samples_after_leading,
        "speech_duration_sec": speech_duration_sec,
        "silence_to_speech_ratio": silence_to_speech_ratio,
        "signal_to_speech_ratio": silence_to_speech_ratio, # Alias retained
    }
    return result


def compute_and_save_global_silence_stats(full_df, target_sr, output_dir, logger, **silence_kwargs):
    """
    Iterates over the entire dataset to calculate and save global silence statistics.
    Saves outputs to the configured output_dir.
    """
    logger.info("Computing per-audio leading silence statistics...")
    records = []
    
    stats_dir = Path(output_dir) / "silence_stats"

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Silence stats"):
        try:
            records.append(silence_stats_for_row(row, target_sr, **silence_kwargs))
        except Exception as e:
            records.append({
                "audio_path": row.get("audio_path", ""),
                "metadata_path": row.get("metadata_path", ""),
                "new_split": row.get("new_split", ""),
                "lang_id": row.get("lang_id", ""),
                "error": repr(e),
            })

    stats_df = pd.DataFrame(records)
    per_audio_path = stats_dir / "global_per_audio_silence_stats.csv"
    stats_df.to_csv(per_audio_path, index=False)

    valid_df = stats_df[pd.to_numeric(stats_df.get("leading_silence_samples"), errors="coerce").notna()].copy()
    if len(valid_df) > 0:
        aggregate_df = (
            valid_df
            .groupby(["new_split", "lang_id"], dropna=False)
            .agg(
                n=("audio_path", "count"),
                mean_leading_silence_samples=("leading_silence_samples", "mean"),
                median_leading_silence_samples=("leading_silence_samples", "median"),
                std_leading_silence_samples=("leading_silence_samples", "std"),
                min_leading_silence_samples=("leading_silence_samples", "min"),
                max_leading_silence_samples=("leading_silence_samples", "max"),
                mean_leading_silence_sec=("leading_silence_sec", "mean"),
                median_leading_silence_sec=("leading_silence_sec", "median"),
                mean_silence_to_speech_ratio=("silence_to_speech_ratio", "mean"),
                median_silence_to_speech_ratio=("silence_to_speech_ratio", "median"),
            )
            .reset_index()
        )
        aggregate_path = stats_dir / "global_aggregate_silence_stats_by_split_lang.csv"
        aggregate_df.to_csv(aggregate_path, index=False)
        logger.info(f"Saved silence stats: {per_audio_path} and {aggregate_path}")
    else:
        logger.warning("No valid silence statistics were computed.")

    return stats_df


def get_pair_avg_ssr(stats_df, filtered_df):
    """Helper to extract the specific SSR for a filtered subset of audio paths."""
    if stats_df is None or len(stats_df) == 0:
        return None
    if "audio_path" not in stats_df.columns:
        return None

    paths = set(filtered_df["audio_path"].astype(str).tolist())
    subset = stats_df[stats_df["audio_path"].astype(str).isin(paths)].copy()
    
    if len(subset) == 0 or "silence_to_speech_ratio" not in subset.columns:
        return None

    values = pd.to_numeric(subset["silence_to_speech_ratio"], errors="coerce")
    values = values[values.replace([float("inf"), -float("inf")], pd.NA).notna()]
    
    if len(values) == 0:
        return None
        
    return float(values.mean())