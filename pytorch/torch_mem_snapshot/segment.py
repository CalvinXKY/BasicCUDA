# Author: kevin.xie  zhihu@kaiyuan

import torch
from datetime import datetime


def segment_example(device="cuda:0"):
    tensor1 = torch.randn(size=(10,1024, 1024, 512), device=device)
    tensor1.to("cpu")
    # free tensor1 ,the segment will be freed as well.
    del tensor1
    torch.cuda.empty_cache()
    # create a new segment and a new block for tensor2
    tensor2 = torch.rand(size=(1, 1024, 512), device=device)
    
    tensor_group = []
    for _ in range(10):
        tensor_group.append(torch.rand(size=(1024, 1024, 512), device=device))


def run():
    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # example running:
    segment_example()

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    run()

