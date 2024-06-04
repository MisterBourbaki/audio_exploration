import torch
from einops import rearrange
from rich import print as pprint
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as sisnr,
    signal_noise_ratio as snr
)

from audio_exploration.kmeans import KMeans, KMeans_cosine, KmeansCodebook, quantize
from audio_exploration.kmeans_from_vq import kmeans


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
        # class_labels, centroids = KMeans(data_train, num_clusters)
        # preds = rearrange(
        #     [centroids[label] for label in class_labels],
        #     "N D -> N D",
        # )
        codebook = KmeansCodebook(dim_embed=dim_embed, num_clusters=num_clusters)
        class_labels = codebook.init_codebook(data_train=data_train)
        # preds = rearrange(
        #     [centroids[label] for label in class_labels],
        #     "N D -> N D",
        # )
        preds = codebook.codebook(class_labels)
        pprint(f"Preds are of shape {preds.shape}")
        sisnr_keops = sisnr(preds, data_train)
        snr_keops = snr(preds, data_train)
        pprint(f"The SI SNR ratio is {sisnr_keops.mean()}")
        pprint(f"The SNR ratio is {snr_keops.mean()}")
        preds, _, _ = codebook(data_val)
        sisnr_keops_val = sisnr(preds, data_val)
        snr_keops_val = snr(preds, data_val)
        pprint(f"The val SI SNR ratio is {sisnr_keops_val.mean()}")
        pprint(f"The val SNR ratio is {snr_keops_val.mean()}")

        data_train = rearrange(data_train, "N D -> 1 N D")
        class_labels_bis, centroids_bis, buckets = kmeans(
            data_train, num_clusters=num_clusters
        )
        pprint(
            f"The shape of class_labels is {class_labels_bis.shape} and of centroids is {centroids_bis.shape} and buckets {buckets.shape}"
        )
        pprint(f"The sum of bins is {centroids_bis.sum()}")
        labels = rearrange(buckets, "1 N -> N")
        centroids = rearrange(class_labels_bis, " 1 N D -> N D")
        preds = rearrange(
            [centroids[label] for label in labels],
            "N D -> N D",
        )
        sisnr_old = sisnr(preds, rearrange(data_train, " 1 N D -> N D"))
        snr_old = snr(preds, rearrange(data_train, " 1 N D -> N D"))
        pprint(f"The SISNR for old algo is {sisnr_old.mean()}")
        pprint(f"The SNR for old algo is {snr_old.mean()}")
        preds, _, _ = quantize(data_val, centroids)
        sisnr_old_val = sisnr(preds, data_val)
        snr_old_val = snr(preds, data_val)
        pprint(f"The val SI SNR ratio is {sisnr_old_val.mean()}")
        pprint(f"The val SNR ratio is {snr_old_val.mean()}")
    else:
        class_labels, centroids = KMeans_cosine(data_train, num_clusters)

    pprint(f"The shapes are {class_labels.shape} and {centroids.shape}")
    pprint(f"For instance, the label for the first data_train is {class_labels[0]}")
