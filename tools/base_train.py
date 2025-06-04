import torch
import argparse
import os
import time


def stress_gpu(gpu_index=0, memory_gb=4, compute_intensity=True):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {gpu_index}")

    # 显存占用
    print(f"Allocating ~{memory_gb} GB on GPU {gpu_index}...")
    tensor_list = []
    alloc_size = int((memory_gb * 1024**3) / 4)  # float32 -> 4 bytes
    try:
        t = torch.empty(alloc_size, dtype=torch.float32, device=device)
        tensor_list.append(t)
        # input()
    except RuntimeError as e:
        print("显存分配失败：", e)
    input('显存占用中')
    # 算力占用
    if compute_intensity:
        print("开始算力占用（持续运算）...")
        a = torch.randn((2048, 2048), device=device)
        b = torch.randn((2048, 2048), device=device)

        while True:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stress Test Tool")
    parser.add_argument('--gpu', type=str, default='0', help='GPU index to use')
    parser.add_argument('--epoch', type=float, default=20, help='Amount of GPU memory to allocate (in GB)')
    parser.add_argument('--compute', default=4.0, action='store_true', help='Enable compute load (matrix multiplication)')

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    stress_gpu(memory_gb=args.epoch, compute_intensity=args.compute)
