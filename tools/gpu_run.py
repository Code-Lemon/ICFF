import pynvml
import subprocess
import time
import os
import argparse

MEM_THRESHOLD_MB = 5000
launched_gpus = set()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def check_and_launch(target_gpus, max_cards, mem):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    status_lines = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mem_mb = mem_info.used / 1024**2

        if i in target_gpus:
            if (
                    used_mem_mb < MEM_THRESHOLD_MB and
                    i not in launched_gpus and
                    len(launched_gpus) < max_cards
            ):
                subprocess.Popen([
                    "python", "./tools/base_train.py",
                    "--gpu", str(i),
                    "--epoch", mem,
                ])
                launched_gpus.add(i)
            mark = "🟩 OCCUPIED" if i in launched_gpus else "🟥 FREE"
        else:
            mark = "🟦 SKIPPED"

        status_lines.append(f"GPU {i}: {used_mem_mb:.1f} MB used\t{mark}")

    pynvml.nvmlShutdown()
    return status_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Auto Occupy Monitor")
    parser.add_argument('--target', type=int, nargs='*',
                        help="GPU indices to monitor and occupy (e.g. --target 0 1 2). If not set, all GPUs are used.")
    parser.add_argument('--max_cards', type=int, default=8,
                        help="Maximum number of GPUs to occupy (default: 8)")
    parser.add_argument('--epoch', type=str, default='20')
    args = parser.parse_args()

    # 自动设置为所有GPU
    pynvml.nvmlInit()
    all_gpu_ids = set(range(pynvml.nvmlDeviceGetCount()))
    pynvml.nvmlShutdown()
    target_gpus = set(args.target) if args.target is not None and len(args.target) > 0 else all_gpu_ids

    try:
        while True:
            lines = check_and_launch(target_gpus, args.max_cards, args.epoch)
            clear_screen()
            print("=== GPU Memory Monitor (Target GPUs: {}) ===".format(','.join(map(str, sorted(target_gpus)))))
            for line in lines:
                print(line)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n监控已终止。")


#
# import pynvml
# import subprocess
# import time
# import os
# import sys
#
# MEM_THRESHOLD_MB = 5000
# launched_gpus = set()
#
# def clear_screen():
#     # 清空控制台
#     os.system('cls' if os.name == 'nt' else 'clear')
#
# def check_and_launch():
#     pynvml.nvmlInit()
#     device_count = pynvml.nvmlDeviceGetCount()
#     status_lines = []
#
#     for i in range(device_count):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         used_mem_mb = mem_info.used / 1024**2
#
#         if used_mem_mb < MEM_THRESHOLD_MB and i not in launched_gpus:
#             subprocess.Popen([
#                 "python", "./tools/base_train.py",
#                 "--gpu", str(i),
#                 "--mem", "19",
#             ])
#             launched_gpus.add(i)
#
#         mark = "🟩 OCCUPIED" if i in launched_gpus else "🟥 FREE"
#         status_lines.append(f"GPU {i}: {used_mem_mb:.1f} MB used\t{mark}")
#
#     pynvml.nvmlShutdown()
#     return status_lines
#
# if __name__ == "__main__":
#     try:
#         while True:
#             lines = check_and_launch()
#             clear_screen()
#             print("=== GPU Memory Monitor ===")
#             for line in lines:
#                 print(line)
#             time.sleep(2)
#     except KeyboardInterrupt:
#         print("\n监控已终止。")
