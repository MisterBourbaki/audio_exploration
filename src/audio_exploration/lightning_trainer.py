from lightning.pytorch.cli import LightningCLI

from vector_quantization.lit_vq import LitVQ
from audio_exploration.data import LightningMFCC

def cli_main():
    cli = LightningCLI(model_class=LitVQ, datamodule_class=LightningMFCC)

if __name__ == "__main__":
    cli_main()