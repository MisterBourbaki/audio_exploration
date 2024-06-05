
from jsonargparse import CLI

from audio_exploration.benchmark import run_bench


def main():
    print("Hello from audio-exploration!")

    CLI(run_bench)
