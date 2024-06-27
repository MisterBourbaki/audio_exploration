from lightning.pytorch import LightningDataModule
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from audio_exploration.extract_features import ConfigMFCC, ConfigMel
from torchaudio.transforms import MFCC
from torchaudio import load

BASE_SAMPLE_RATE = 16000

base_config_mfcc = ConfigMFCC(
    sample_rate=BASE_SAMPLE_RATE,
    n_mfcc=12,
    melkwargs=ConfigMel(
        n_fft=2 * BASE_SAMPLE_RATE // 100,
        n_mels=12,
        hop_length=BASE_SAMPLE_RATE // 100,
        win_length=BASE_SAMPLE_RATE // 100,
        mel_scale="htk",
    ),
)


class MFCCDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        ext: str,
        start_mfcc: int = 0,
        end_mfcc: int = 11,
        duration: int = 1,
        config_mfcc: ConfigMFCC = base_config_mfcc,
    ) -> None:
        super().__init__()
        assert (
            config_mfcc.n_mfcc >= (end_mfcc - start_mfcc) + 1
        ), f"The configuration provided extract {config_mfcc.n_mfcc} MFCC, but you want MFCC from {start_mfcc} index to {end_mfcc} index."
        self.all_files = list(data_dir.glob(f"*{ext}"))
        _, sr = load(self.all_files[0])
        assert (
            sr == config_mfcc.sample_rate
        ), f"The sample rate provided in the MFCC config is {config_mfcc.sample_rate}, but the sample rate for the audio files is {sr}"
        
        self.sr = sr
        self.length = len(self.all_files)
        self.start_mfcc = start_mfcc
        self.end_mfcc = end_mfcc
        self.duration = duration
        self.extract_mfcc = MFCC(**config_mfcc.model_dump())

    def __getitem__(self, index):
        file = self.all_files[index]
        waveform, sr = load(file)
        mfcc = self.extract_mfcc(waveform[:, self.duration * self.sr])
        return mfcc[:, self.start_mfcc : (self.end_mfcc + 1)]
    
    def __len__(self):
        return self.length


class LightningMFCC(LightningDataModule):
    def __init__(
        self,
        data_dir_train: Path,
        data_dir_val: Path,
        data_dir_test: Optional[Path] = None,
        ext: str = ".flac",
        start_mfcc: int = 0,
        end_mfcc: int = 11,
        duration: int = 1,
        config_mfcc: ConfigMFCC = base_config_mfcc,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.data_dir_train = data_dir_train
        self.data_dir_val = data_dir_val
        self.data_dir_test = data_dir_test
        self.ext = ext
        self.start_mfcc = start_mfcc
        self.end_mfcc = end_mfcc
        self.duration = duration
        self.config_mfcc = config_mfcc
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = MFCCDataset(data_dir=self.data_dir_train, ext=self.ext, start_mfcc=self.start_mfcc, end_mfcc=self.start_mfcc, duration=self.duration, config_mfcc=self.config_mfcc)
            self.val_set = MFCCDataset(data_dir=self.data_dir_val, ext=self.ext, start_mfcc=self.start_mfcc, end_mfcc=self.start_mfcc, duration=self.duration, config_mfcc=self.config_mfcc)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)