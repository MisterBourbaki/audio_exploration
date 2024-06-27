from pathlib import Path
from typing import Callable, Optional

import einops
import torch
import torchaudio
from pydantic import BaseModel
from rich import print as pprint
from torchaudio.functional import detect_pitch_frequency
from torchaudio.transforms import LFCC, MFCC




class ConfigMel(BaseModel):
    n_fft: int = 400
    win_length: Optional[int] = None
    hop_length: Optional[int] = None
    f_min: float = 0.0
    f_max: Optional[float] = None
    pad: int = 0
    n_mels: int = 128
    window_fn: Callable[..., torch.Tensor] = torch.hann_window
    power: float = 2.0
    normalized: bool = False
    wkwargs: Optional[dict] = None
    center: bool = True
    pad_mode: str = "reflect"
    onesided: Optional[bool] = None
    norm: Optional[str] = None
    mel_scale: str = "htk"


class ConfigMFCC(BaseModel):
    sample_rate: int
    n_mfcc: int
    dct_type: int = 2
    norm: str = "ortho"
    log_mels: bool = False
    melkwargs: ConfigMel


def extract_batch_mfcc(
    data_dir: Path,
    conf: ConfigMFCC,
    ext: str = ".flac",
    duration: int = 1,
):
    all_files = list(data_dir.glob(f"**/*{ext}"))
    pprint(f"There are {len(all_files)} files")
    samples = [torchaudio.load(file)[0] for file in all_files]
    sample_rate = torchaudio.load(all_files[0])[1]
    same_duration_samples, ps = einops.pack(
        [sample[:, : duration * sample_rate] for sample in samples], "* time"
    )
    mfcc_transform = MFCC(**conf.model_dump())
    batch_mfcc = mfcc_transform(same_duration_samples)
    pprint(f"Batch mfcc shape is {batch_mfcc.shape}")
    return batch_mfcc




if __name__ == "__main__":
    data_dir = Path("/home/mrbourbaki/Work/data/audio/LibriSpeech/test-clean/61/70968/")
    data_dir_vctk = Path(
        "/home/mrbourbaki/Work/data/audio/VCTK-Corpus-0.92/wav48_silence_trimmed/s5"
    )
    SAMPLE_SPEECH = data_dir / Path("61-70968-0000.flac")
    SAMPLE_SPEECH_VCTK = data_dir_vctk / Path("s5_001_mic1.flac")
    samples = [torchaudio.load(file)[0] for file in data_dir.glob("*.flac")]
    metadata_samples = [
        torchaudio.info(file, format="flac") for file in data_dir.glob("*.flac")
    ]

    pprint(f"There are {len(samples)} samples")
    for metadata in metadata_samples:
        pprint(f"Here are the metadata for all samples {metadata}")

    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)
    pprint(f"The shape of SPEECH_WAVEFORM is {SPEECH_WAVEFORM.shape}")
    SPEECH_WAVEFORM_VCTK, SAMPLE_RATE_VCTK = torchaudio.load(SAMPLE_SPEECH_VCTK)
    pprint(
        f"Sample rates are {SAMPLE_RATE} for LibrichSpeech and {SAMPLE_RATE_VCTK} for VCTK"
    )

    sample_rate = SAMPLE_RATE
    n_fft = 2 * SAMPLE_RATE // 100
    win_length = SAMPLE_RATE // 100
    hop_length = SAMPLE_RATE // 100
    n_mels = 12
    n_mfcc = 12

    config_mfcc = ConfigMFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs=ConfigMel(
            n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, mel_scale="htk"
        ),
    ).model_dump()

    mfcc_transform = MFCC(**config_mfcc)

    lfcc_transform = LFCC(
        sample_rate=sample_rate,
        n_lfcc=256,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )

    mfcc = mfcc_transform(SPEECH_WAVEFORM)
    lfcc = lfcc_transform(SPEECH_WAVEFORM)

    pitch = detect_pitch_frequency(SPEECH_WAVEFORM, sample_rate=sample_rate)

    pprint(f"The sample rate is {sample_rate}")
    pprint(f"Shape of MFCC are {mfcc.shape}")
    pprint(f"Shape of LFCC are {lfcc.shape}")
    pprint(f"Shape of pitch are {pitch.shape}")

    batch_mfcc = [mfcc_transform(sample) for sample in samples]
    pprint(f"Shape of MFCC are {[mfcc.shape for mfcc in batch_mfcc]}")

    one_second_samples, ps = einops.pack(
        [sample[:, :SAMPLE_RATE] for sample in samples], "* time"
    )
    pprint(f"Shape of batch is {one_second_samples.shape}")
    batch_mfcc = mfcc_transform(one_second_samples)
    pprint(f"Batch mfcc shape is {batch_mfcc.shape}")

    config_mfcc = ConfigMFCC(
        sample_rate=sample_rate,
        n_mfcc=12,
        melkwargs=ConfigMel(
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            mel_scale="htk",
        ),
    )

    mfcc = extract_batch_mfcc(data_dir=data_dir_vctk, conf=config_mfcc)
    pprint(mfcc.shape)
