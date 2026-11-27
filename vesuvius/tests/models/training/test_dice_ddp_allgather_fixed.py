"""Compare batch Dice losses and model gradients against full-batch training."""

from __future__ import annotations

import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from vesuvius.models.training.loss.nnunet_losses import (
    MemoryEfficientSoftDiceLoss,
    SoftDiceLoss,
)


requires_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)

WORLD = 2
WORKER_TIMEOUT_S = 120
LOSSES = {"soft": SoftDiceLoss, "memory_efficient": MemoryEfficientSoftDiceLoss}


def _batch():
    generator = torch.Generator().manual_seed(0)
    probabilities = torch.rand(4, 3, 8, 8, generator=generator)
    labels = torch.randint(0, 3, (4, 1, 8, 8), generator=generator)
    return probabilities, labels


def _loss(name, ddp):
    return LOSSES[name](apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.0, ddp=ddp)


def _rank_half(rank, tensor):
    half = tensor.shape[0] // WORLD
    return tensor[rank * half:(rank + 1) * half]


def _model():
    torch.manual_seed(42)
    return torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)


def _worker(rank, name, init_file, out_prefix):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    try:
        probabilities, labels = _batch()
        model = DistributedDataParallel(_model())
        local = model(_rank_half(rank, probabilities)).softmax(dim=1)
        loss = _loss(name, ddp=True)(local, _rank_half(rank, labels))
        loss.backward()
        torch.save({"loss": loss.detach(), "grad": model.module.weight.grad}, f"{out_prefix}.{rank}")
    finally:
        dist.destroy_process_group()
    # Leave without interpreter shutdown. The gloo process group outlives
    # destroy_process_group() and is freed during finalization, where Python
    # 3.14 parks any gloo thread that asks for the GIL while it holds the
    # group's lock; ~ProcessGroupGloo then waits on that lock forever and
    # mp.spawn(join=True) never returns. The result is already on disk.
    os._exit(0)


@requires_gloo
@pytest.mark.parametrize("name", sorted(LOSSES))
def test_batch_dice_under_ddp_matches_one_process(tmp_path, name):
    probabilities, labels = _batch()
    model = _model()
    reference = _loss(name, ddp=False)(model(probabilities).softmax(dim=1), labels)
    reference.backward()

    out_prefix = str(tmp_path / "rank")
    context = mp.spawn(
        _worker,
        args=(name, str(tmp_path / "process_group"), out_prefix),
        nprocs=WORLD,
        join=False,
    )
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail(f"DDP workers did not exit within {WORKER_TIMEOUT_S} s")

    for rank in range(WORLD):
        result = torch.load(f"{out_prefix}.{rank}")
        torch.testing.assert_close(result["loss"], reference.detach())
        torch.testing.assert_close(result["grad"], model.weight.grad)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_batch_dice_without_process_group(name):
    probabilities, labels = _batch()
    probabilities.requires_grad_(True)
    reference = _loss(name, ddp=False)(probabilities, labels)
    actual = _loss(name, ddp=True)(probabilities, labels)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, probabilities), torch.autograd.grad(reference, probabilities)
    )
