import time

import torch
from einops import rearrange
from pykeops.torch import LazyTensor
from rich import print as pprint
from rich.progress import track
from torch.nn import Module, Embedding

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

def quantize(input, centroids):
    x_i = LazyTensor(rearrange(input, " N D -> N 1 D"))  # (N, 1, D) samples
    centroids_j = LazyTensor(
        rearrange(centroids, "K D -> 1 K D")
    )  # (1, K, D) centroids
    distances_ij = ((x_i - centroids_j) ** 2).sum(
        -1
    )  # (N, K) symbolic squared distances
    class_labels = (
        distances_ij.argmin(dim=1).long().view(-1)
    )  # Points -> Nearest cluster
    preds = rearrange(
            [centroids[label] for label in class_labels],
            "N D -> N D",
        )

    return preds, class_labels, distances_ij

class KmeansCodebook(Module):
    def __init__(self, dim_embed, num_clusters) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.num_clusters = num_clusters
        
    def init_codebook(self, data_train, num_iters: int = 10):
        class_labels, centroids = KMeans(x=data_train, K=self.num_clusters, Niter=num_iters)
        self.codebook = Embedding(num_embeddings=self.num_clusters, embedding_dim=self.dim_embed).from_pretrained(embeddings=centroids)
        return class_labels

    def forward(self, input):
        x_i = LazyTensor(rearrange(input, " N D -> N 1 D"))  # (N, 1, D) samples
        centroids_j = LazyTensor(
            rearrange(self.codebook.weight, "K D -> 1 K D")
        )  # (1, K, D) centroids
        distances_ij = ((x_i - centroids_j) ** 2).sum(
            -1
        )  # (N, K) symbolic squared distances
        class_labels = (
            distances_ij.argmin(dim=1).long().view(-1)
        )  # Points -> Nearest cluster
        preds = self.codebook(class_labels)

        return preds, class_labels, distances_ij


def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    centroids = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    centroids_j = LazyTensor(centroids.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - class_labels is the (N,) vector of class labels
    # - centroids  is the (K, D) cloud of cluster centroids
    for _ in track(range(Niter)):
        # E step: assign points to the closest cluster -------------------------
        distances_ij = ((x_i - centroids_j) ** 2).sum(
            -1
        )  # (N, K) symbolic squared distances
        class_labels = (
            distances_ij.argmin(dim=1).long().view(-1)
        )  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, class_labels[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        num_points_per_cluster = (
            torch.bincount(class_labels, minlength=K).type_as(centroids).view(K, 1)
        )
        centroids /= num_points_per_cluster  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        pprint(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        pprint(
            f"Timing for {Niter} iterations: {end - start:.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n"
        )

    return class_labels, centroids


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    centroids = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    centroids = torch.nn.functional.normalize(centroids, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    centroids_j = LazyTensor(centroids.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - class_labels is the (N,) vector of class labels
    # - centroids  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | centroids_j  # (N, K) symbolic Gram matrix of dot products
        class_labels = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, class_labels[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        centroids[:] = torch.nn.functional.normalize(centroids, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        pprint(
            f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        pprint(
            f"Timing for {Niter} iterations: {end - start:.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n"
        )

    return class_labels, centroids
