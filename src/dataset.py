import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# Import the audio processing helpers we built in utils.py
from src.utils import (
    load_audio_mono_resample, 
    compute_leading_silence_samples, 
    crop_or_pad
)

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class FleursLIDDataset(Dataset):
    def __init__(
        self,
        df,
        max_duration_sec,
        target_sr,
        trim_leading=False,
        return_metadata=False,
        silence_kwargs=None
    ):
        """
        Args:
            df (pd.DataFrame): The metadata dataframe for this split.
            max_duration_sec (float): Target audio length in seconds.
            target_sr (int): Target sampling rate.
            trim_leading (bool): If True, dynamically strips leading silence.
            return_metadata (bool): If True, returns paths and raw values for analysis.
            silence_kwargs (dict): Parameters for the silence detector (db_threshold, etc.)
        """
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.trim_leading = trim_leading
        self.return_metadata = return_metadata
        self.silence_kwargs = silence_kwargs if silence_kwargs is not None else {}

        # Standard MFCC Extractor for FLEURS
        self.mfcc_transform = T.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=60,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 60,
                "center": False,
            },
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = int(row["label"])

        # 1. Load and Resample
        waveform, sr = load_audio_mono_resample(audio_path, target_sr=self.target_sr)

        # 2. Silence Detection
        leading_silence_samples = compute_leading_silence_samples(
            waveform, 
            sample_rate=sr, 
            **self.silence_kwargs
        )

        # 3. Dynamic Trimming
        if self.trim_leading:
            waveform = waveform[:, leading_silence_samples:]

        # 4. Length Enforcement & Feature Extraction
        waveform = crop_or_pad(waveform, self.max_samples)
        mfcc = self.mfcc_transform(waveform)

        # 5. Return Formatting
        if self.return_metadata:
            return (
                mfcc,
                torch.tensor(label, dtype=torch.long),
                str(audio_path),
                str(row.get("metadata_path", "")),
                int(row.get("lang_id", -1)),
                int(leading_silence_samples),
            )

        return mfcc, torch.tensor(label, dtype=torch.long)


# ==========================================
# 2. DATALOADER FACTORY
# ==========================================
def make_loader(
    df, 
    duration, 
    trim_leading, 
    batch_size, 
    shuffle, 
    num_workers, 
    target_sr, 
    return_metadata=False, 
    silence_kwargs=None
):
    """
    Helper function to instantly spin up a DataLoader with the correct settings.
    """
    dataset = FleursLIDDataset(
        df=df,
        max_duration_sec=duration,
        target_sr=target_sr,
        trim_leading=trim_leading,
        return_metadata=return_metadata,
        silence_kwargs=silence_kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )