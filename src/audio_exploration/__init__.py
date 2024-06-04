import sys

import torch
from einops import rearrange
from jsonargparse import CLI
from rich import print as pprint
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as sisnr

import audio_exploration
from audio_exploration.benchmark import run_bench
from audio_exploration.kmeans import KMeans, KMeans_cosine
from audio_exploration.kmeans_from_vq import kmeans


def main():
    print("Hello from audio-exploration!")

    CLI(run_bench)
