import torch
from rich import print as pprint
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio import (
    signal_noise_ratio 
)
from audio_exploration.vector_quantizer import VectorQuantizer


class RandomData(Dataset):
    def __init__(self, dim: int, num_samples: int, mean: float = 0.3, var: float = 0.7) -> None:
        super().__init__()
        self.dim = dim
        self.num_samples = num_samples
        self.samples = var * torch.randn(num_samples, self.dim) + mean

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.num_samples


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch_idx, input in enumerate(dataloader):
        # Compute prediction and loss
        pred, vq_loss = model(input)
        loss = vq_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * dataloader.batch_size + len(input)
            pprint(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    snr = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for input in dataloader:
            pred, vq_loss = model(input)
            test_loss += vq_loss.item()
            snr += signal_noise_ratio(pred, input).mean()
            # pprint(snr.shape)

    test_loss /= num_batches
    snr /= num_batches
    pprint(
        f"Test Error: \n SNR: {(snr):>0.1f}, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    learning_rate = 1e-2
    batch_size = 1000
    epochs = 500

    num_embeddings = 32
    embedding_dim = 2
    num_samples = 50000

    model = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(
        RandomData(dim=embedding_dim, num_samples=num_samples), batch_size=batch_size
    )
    test_dataloader = DataLoader(
        RandomData(dim=embedding_dim, num_samples=num_samples), batch_size=batch_size
    )

    for t in range(epochs):
        pprint(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model)
    pprint("Done!")
