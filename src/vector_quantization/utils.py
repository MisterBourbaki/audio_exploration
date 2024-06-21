
import torch
from aenum import Enum
from pykeops.torch import LazyTensor
from torch.nn import functional


def compute_pairwise_distances(
    input: torch.Tensor, target: torch.Tensor, return_score: bool = False
) -> torch.Tensor:
    """Compute pairwise distances between tensors input and target.

    Parameters
    ----------
    input : torch.Tensor
        expected shape (N, D)
    target : torch.Tensor
        expected shape (K, D)

    Returns
    -------
    torch.Tensor
        tensor of shape (N, K)
    """
    distances = (
        torch.sum(input**2, dim=1, keepdim=True)
        + torch.sum(target**2, dim=1)
        - 2 * torch.matmul(input, target.t())
    )
    if return_score:
        distances = 1 / (distances + 1e-8)
    return distances


def compute_pairwise_distances_pykeops(
    input: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise distances between tensors input and target.

    Parameters
    ----------
    input : torch.Tensor
        expected shape (N, D)
    target : torch.Tensor
        expected shape (K, D)

    Returns
    -------
    torch.Tensor
        tensor of shape (N, K)
    """
    N, D = input.shape
    K, D = target.shape
    input_i = LazyTensor(input.view(N, 1, D))
    target_j = LazyTensor(target.view(1, K, D))
    distances = ((input_i - target_j) ** 2).sum(-1)
    return distances


class SampleMethods(Enum):
    """Enumeration of sample methods.

    Attributes
    ----------
    STGS:
        Straight Through Gumble Sampling method.
    GS:
        Gumbel Sampling.
    REINMAX:
        the reinmax sampling method, a second order Straight Through algo.
    ST:
        the classical Straight Through method.

    """

    STGS = "STGS"
    GS = "GS"
    REINMAX = "reinmax"
    ST = "ST"


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1,
    method: SampleMethods = SampleMethods.STGS,
    dim: int = -1,
) -> torch.Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      temperature: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, temperature=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, temperature=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,temperature)
    y_soft = gumbels.softmax(dim)

    if method is SampleMethods.REINMAX:
        π0 = logits.softmax(dim=dim)
        dtype, size = logits.dtype, logits.shape[dim]
        ind = logits.argmax(dim=dim)
        one_hot = functional.one_hot(ind, size).type(dtype)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((torch.log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
        ret = one_hot

    elif method is SampleMethods.STGS:
        # Straight through with Gumbel sampling.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    elif method is SampleMethods.GS:
        # Reparametrization trick.
        ret = y_soft
    elif method is SampleMethods.ST:
        raise "Not implemented yet!"
    else:
        raise "No methods with that name!"

    return ret
