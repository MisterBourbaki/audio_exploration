from lightning.pytorch import LightningModule
from torch.nn import Module, MSELoss
from torch.optim import AdamW

from src.vector_quantization.vector_quantizer import VectorQuantizer


class LitVQ(LightningModule):
    def __init__(
        self,
        encoder_module: Module,
        decoder_module: Module,
        num_embeddings,
        embedding_dim,
        beta,
        lr,
    ) -> None:
        super().__init__()
        self.encoder = encoder_module
        self.decoder = decoder_module

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.reconstruction_loss = MSELoss()
        self.lr = lr

        self.quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            beta=self.beta,
        )

    def training_step(self, batch):
        features = self.encoder(batch)
        quantized, vq_loss = self.quantizer(features)
        preds = self.decoder(quantized)
        recon_loss = self.reconstruction_loss(features, preds)
        total_loss = recon_loss + vq_loss

        self.log("train_vq_loss", vq_loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch):
        features = self.encoder(batch)
        quantized, vq_loss = self.quantizer(features)
        preds = self.decoder(quantized)
        recon_loss = self.reconstruction_loss(features, preds)
        total_loss = recon_loss + vq_loss

        self.log("val_vq_loss", vq_loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_total_loss", total_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
