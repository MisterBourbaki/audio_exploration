import torch
from einops import rearrange
from rich import print as pprint
from torch.nn.functional import mse_loss
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as sisnr,
)
from torchmetrics.functional.audio import (
    signal_noise_ratio as snr,
)

from audio_exploration.kmeans import KMeans_cosine, KmeansQuantizer, quantize
from audio_exploration.kmeans_from_vq import kmeans, kmeans_improved

def validation(model, data_val):
    preds, _, _ = model(data_val)
    sisnr_values = sisnr(preds, data_val)
    snr_values = snr(preds, data_val)
    distortion = mse_loss(data_val, preds)
    return {"sisnr": sisnr_values, "snr": snr_values, "distortion": distortion}

def run_bench(
    num_samples: int = 10000,
    dim_embed: int = 2,
    num_clusters: int = 50,
    use_cosine: bool = False,
):
    # num_samples, dim_embed, num_clusters = 100000, 5, 500
    # use_cosine: bool = False

    # Define our dataset:

    data_train = 0.7 * torch.randn(num_samples, dim_embed) + 0.3
    data_val = 0.7 * torch.randn(num_samples, dim_embed) + 0.3

    # Perform the computation:

    if not use_cosine:
        quantizer = KmeansQuantizer(dim_embed=dim_embed, num_clusters=num_clusters)
        class_labels = quantizer.init_codebook(data_train=data_train)
        
        metrics = validation(quantizer, data_val)
        pprint(f"The val SI SNR ratio is {metrics['sisnr'].mean()}")
        pprint(f"The val SNR ratio is {metrics['snr'].mean()}")
        pprint(f"The val distortion is {metrics['distortion']}")

        data_train = rearrange(data_train, "N D -> 1 N D")
        class_labels_bis, centroids_bis, buckets = kmeans(
            data_train, num_clusters=num_clusters
        )
        centroids = rearrange(class_labels_bis, " 1 N D -> N D")
        
        preds, _, _ = quantize(data_val, centroids)
        sisnr_old_val = sisnr(preds, data_val)
        snr_old_val = snr(preds, data_val)
        distortion = mse_loss(data_val, preds)
        pprint(f"The val SI SNR ratio is {sisnr_old_val.mean()}")
        pprint(f"The val SNR ratio is {snr_old_val.mean()}")
        pprint(f"The val distortion is {distortion}")

        data_train = rearrange(data_train, "1 N D -> N D")
        class_labels_improved, centroids_improved = kmeans_improved(
            data_train, num_clusters=num_clusters
        )
        
        preds = rearrange(
            [centroids_improved[label] for label in class_labels_improved],
            "N D -> N D",
        )
        sisnr_improved = sisnr(preds, data_train)
        snr_improved = snr(preds, data_train)
        pprint(f"The SISNR for improved algo is {sisnr_improved.mean()}")
        pprint(f"The SNR for improved algo is {snr_improved.mean()}")
        preds, _, _ = quantize(data_val, centroids_improved)
        sisnr_improved_val = sisnr(preds, data_val)
        snr_improved_val = snr(preds, data_val)
        distortion = mse_loss(data_val, preds)
        pprint(f"The val SI SNR ratio is {sisnr_improved_val.mean()}")
        pprint(f"The val SNR ratio is {snr_improved_val.mean()}")
        pprint(f"The val distortion is {distortion}")
    else:
        class_labels, centroids = KMeans_cosine(data_train, num_clusters)

    pprint(f"The shapes are {class_labels.shape} and {centroids.shape}")
    pprint(f"For instance, the label for the first data_train is {class_labels[0]}")
