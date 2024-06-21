from lightning.pytorch.cli import LightningCLI

from vector_quantization.lit_vq import LitVQ

def cli_main():
    cli = LightningCLI(model_class=LitVQ)

if __name__ == "__main__":
    cli_main()