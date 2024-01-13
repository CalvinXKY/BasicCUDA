# Author: kevin.xie  zhihu@kaiyuan

import torch
from torch import nn
from datetime import datetime


def train(num_iter=5, device="cuda:0"):
    model = nn.Transformer(d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device=device)
    x = torch.randn(size=(1, 1024, 512), device=device)
    tgt = torch.rand(size=(1, 1024, 512), device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(num_iter):
        y = model(x, tgt)
        loss = criterion(y, labels)
        loss.backward()
        print(loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def run():
    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # training running:
    train()

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    run()

