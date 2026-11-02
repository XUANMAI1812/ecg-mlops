"""Benchmark ResNet1D training steps on PTB-XL."""

import argparse
import resource
import time

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg.data.dataset import ECGDataset
from ecg.models.resnet1d import ECGResNet1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ptb-xl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cpu")
    process = psutil.Process()

    dataset = ECGDataset(
        args.data_dir,
        folds=tuple(range(1, 9)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ECGResNet1D().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Device: {device}")
    print(f"Train records: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Requested batches: {args.num_batches}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Initial RAM: {process.memory_info().rss / 1024**2:.1f} MB")

    batch_times = []
    start_total = time.perf_counter()
    model.train()

    for batch_index, (signals, labels) in enumerate(loader):
        if batch_index >= args.num_batches:
            break

        batch_start = time.perf_counter()

        signals = signals.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)

        if (batch_index + 1) % 5 == 0:
            ram_mb = process.memory_info().rss / 1024**2
            print(
                f"Batch {batch_index + 1} | "
                f"time={batch_time:.2f}s | RAM={ram_mb:.1f} MB"
            )

    total_time = time.perf_counter() - start_total
    final_ram = process.memory_info().rss / 1024**2
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Batches completed: {len(batch_times)}")
    print(f"Total time: {total_time:.2f}s")

    if batch_times:
        print(
            "Average training step: "
            f"{sum(batch_times) / len(batch_times):.2f}s"
        )

    print(f"Final RAM: {final_ram:.1f} MB")
    print(f"Max RSS: {max_rss:.1f} MB")


if __name__ == "__main__":
    main()
