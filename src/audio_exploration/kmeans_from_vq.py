import time

import torch
from einops import rearrange, reduce, repeat
from pyinstrument import Profiler
from torch import distributed, einsum, nn
from torch.nn.functional import normalize

profiler = Profiler()


def noop(*args, **kwargs):
    pass


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (
        (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy)
        .clamp(min=0)
        .sqrt()
    )


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack(
        [sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0
    )


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(
            num, all_num_samples / all_num_samples.sum()
        )
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    use_cosine_sim=False,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=noop,
):
    profiler.start()
    start = time.time()
    num_codebooks, dim, dtype, device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)
    # means = samples

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = normalize(new_means, p=2, dim=-1)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    end = time.time()
    print(f"It took {end - start} time")
    profiler.stop()
    profiler.print()
    return means, bins, buckets


def kmeans_improved(vectors, num_clusters=10, num_iter=10):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    profiler.start()

    K = num_clusters
    _, D = vectors.shape  # Number of samples, dimension of the ambient space

    centroids = vectors[:K, :].clone()  # Simplistic initialization for the centroids

    for _ in range(num_iter):
        # E step: assign points to the closest cluster -------------------------
        distances = (
            torch.sum(vectors**2, dim=1, keepdim=True)
            + torch.sum(centroids**2, dim=1)
            - 2 * torch.matmul(vectors, centroids.t())
        )
        class_labels = distances.argmin(dim=1).long().view(-1)

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, class_labels[:, None].repeat(1, D), vectors)

        # Divide by the number of points per cluster:
        num_points_per_cluster = (
            torch.bincount(class_labels, minlength=K).type_as(centroids).view(K, 1)
        )
        centroids /= num_points_per_cluster  # in-place division to compute the average

    profiler.stop()
    profiler.print()
    return class_labels, centroids
