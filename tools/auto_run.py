import subprocess
import threading
import time
import queue
import GPUtil
import os
import hashlib
from rich.live import Live
from rich.table import Table
import argparse
from rich.text import Text
from threading import Lock
from rich.console import Console
from threading import Event

enable_default_task = Event()  # ✅ 线程安全控制变量
enable_default_task.clear()    # 默认不启用占卡
console = Console()

loaded_commands_set = set()
gpu_lock = Lock()
shared_gpus = []
default_running = {}
default_block_until = {}


def monitor_and_run(args):
    while True:
        with gpu_lock:
            gpus = [gpu for gpu in GPUtil.getGPUs() if gpu.id in shared_gpus]

        for gpu in gpus:
            # GPU 空闲并且未运行任何任务
            if gpu.memoryFree > args.mem * 1024 and gpu.id not in running_commands:

                # ✅ 情况 1：有任务 → 正常调度任务
                if not pending_commands.empty():
                    try:
                        task = pending_commands.get_nowait()

                        log_filename = command_to_filename(task["cmd"])
                        log_path = os.path.join(args.log_dir, log_filename)

                        with open(log_path, 'w') as f:
                            f.write(f"# Command: {task['cmd']}\n\n")

                        screen_name = f"gpu{gpu.id}"
                        env_prefix = f"CUDA_VISIBLE_DEVICES={gpu.id} "

                        full_cmd = (
                            f'screen -S {screen_name} -dm bash -c "'
                            f'source ~/.bashrc && '  # ✅ 可激活conda的环境变量（保险起见加上）
                            f'source {args.conda_path} && '
                            f'conda activate {args.conda_env} && '
                            f'cd {args.project_path} && '
                            f'export PYTHONPATH=$(pwd) && '
                            f'{env_prefix}{task["cmd"]}; '
                            f'echo [INFO] Task finished on GPU {gpu.id}; '
                            f'wait"'
                        )
                        # full_cmd = (
                        #     f"screen -S {screen_name} -dm bash -c '"
                        #     f"source ~/.bashrc && "
                        #     f"source {args.conda_path} && "
                        #     f"conda activate {args.conda_env} && "
                        #     f"cd {args.project_path} && "
                        #     f"export PYTHONPATH=\"$(pwd)\" && "
                        #     f"{{ {env_prefix}{task['cmd']} ; echo \"[INFO] Task finished on GPU {gpu.id}\"; }} "
                        #     f"2>&1 | tee \"{log_path}\"; "
                        #     f"wait'"
                        # )

                        subprocess.Popen(full_cmd, shell=True)
                        running_commands[gpu.id] = {
                            "task": task,
                            "screen": screen_name,
                            "cmd": task["cmd"],
                        }

                        # 如果 GPU 上有默认任务，先终止它
                        if gpu.id in default_running:
                            default_screen = f"default_gpu{gpu.id}"
                            subprocess.run(f"screen -S {default_screen} -X quit", shell=True)
                            del default_running[gpu.id]
                            console.print(f"[INFO] Default task on GPU {gpu.id} terminated to run new task", style="yellow")

                    except queue.Empty:
                        break

                # ✅ 情况 2：无任务且 GPU 空闲 → 启动默认任务（如果未启动）
                elif (
                        enable_default_task.is_set() and
                        gpu.id not in default_running and
                        (gpu.id not in default_block_until or time.time() > default_block_until[gpu.id])
                ):
                    default_cmd = args.default_cmd
                    screen_name = f"default_gpu{gpu.id}"
                    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu.id} "

                    full_cmd = (
                        f'screen -S {screen_name} -dm bash -c "'
                        f'source {args.conda_path} && '
                        f'conda activate {args.conda_env} && '
                        f'cd {args.project_path} && '
                        f'{env_prefix}{default_cmd}"'
                    )

                    subprocess.Popen(full_cmd, shell=True)
                    default_running[gpu.id] = None
                    console.print(f"[INFO] Default task started on GPU {gpu.id}", style="dim")

        time.sleep(args.poll_interval)


def check_completed(args):
    while True:
        # 获取当前所有screen会话列表
        active_screens = subprocess.getoutput("screen -ls")

        for gpu_id, info in list(running_commands.items()):
            screen_name = info["screen"]

            # 如果当前 screen_name 不在活动 screen 列表中，说明它已经结束
            if screen_name not in active_screens:
                completed_commands.append({
                    "cmd": info["cmd"],
                    "log": info.get("log", "-")
                })
                del running_commands[gpu_id]
                print(f"[INFO] Task completed on GPU {gpu_id}: {info['cmd']}")

        time.sleep(args.poll_interval)

def display_status():
    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        while True:
            table = Table(title="GPU & Task Monitor", show_lines=True)
            table.add_column("Type", style="bold cyan", justify="right")
            table.add_column("GPU ID / Status", style="bold magenta")
            table.add_column("Info", style="white")
            table.add_column("completed", style="white")

            with gpu_lock:
                gpus_to_highlight = set(shared_gpus)

            type_list = []
            gpu_id_list = []
            info_list = []
            completed_list = []
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_id = str(gpu.id)
                mem_free = f"{int(gpu.memoryFree)} MB"
                if gpu.id in running_commands:
                    running_cmd = running_commands[gpu.id]["cmd"]
                    row_type = Text("Running", style="bold yellow")
                elif gpu.id in default_running:
                    running_cmd = f"[DEFAULT] {args.default_cmd}"
                    row_type = Text("Default", style="bold red")
                else:
                    running_cmd = "-"
                    row_type = "GPU"
                style = "bold green" if gpu.id in gpus_to_highlight else ""

                type_list.append('GPU')
                gpu_id_list.append(gpu_id)
                info_list.append(Text(f"Free: {mem_free}, Running: {running_cmd}", style=style))

            for task in list(pending_commands.queue):
                type_list.append(Text("Pending", style="bold red"))
                gpu_id_list.append("-")
                info_list.append(task["cmd"])

                # table.add_row(Text("Pending", style="bold red"), "-", task["cmd"])

            # for info in running_commands.values():
            #     table.add_row(Text("Running", style="bold yellow"), "-", info["cmd"])

            for cmd in completed_commands:
                completed_list.append(Text(f"Completed: {cmd['cmd']}", style="bold green"))
            while len(type_list) > len(completed_list):
                completed_list.append('-')
            while len(type_list) < len(completed_list):
                type_list.append("-")
                gpu_id_list.append("-")
                info_list.append("-")
            for i in range(len(type_list)):
                table.add_row(type_list[i], gpu_id_list[i], info_list[i], completed_list[i])

            live.update(table)
            console.print("", end="")
            time.sleep(2)



def gpu_input_listener():
    global shared_gpus
    while True:
        try:
            new_input = console.input(
                "\n[bold cyan][INPUT][/bold cyan] Commands:\n"
                "  [green]gpu 0 1 2[/green]   → Set active GPUs\n"
                "  [red]kill 0[/red]          → Kill task on GPU 0\n"
                "  [cyan]update[/cyan]        → Reload task file and append new tasks\n"
                "  [magenta]hold on/off[/magenta] → Enable/disable default GPU-holding task\n"
                "  [yellow]clear-completed[/yellow] → Clear completed tasks\n"
                "> "
            ).strip()

            if not new_input:
                continue

            parts = new_input.strip().split()
            command = parts[0].lower()

            if command == "gpu":
                if len(parts) < 2:
                    console.print("[ERROR] Usage: gpu <gpu_id1> <gpu_id2> ...", style="bold red")
                    continue
                new_gpu_ids = list(map(int, parts[1:]))
                with gpu_lock:
                    shared_gpus.clear()
                    shared_gpus.extend(new_gpu_ids)
                console.print(f"[INFO] GPU list updated to: {shared_gpus}", style="green")

            elif command == "kill":
                if len(parts) != 2 or not parts[1].isdigit():
                    console.print("[ERROR] Usage: kill <gpu_id>", style="bold red")
                    continue
                gpu_id = int(parts[1])

                if gpu_id in running_commands:
                    screen_name = running_commands[gpu_id]["screen"]
                    console.print(f"[INFO] Killing screen: {screen_name} (GPU {gpu_id})...", style="bold red")
                    subprocess.run(f"screen -S {screen_name} -X quit", shell=True)
                    killed_cmd = running_commands[gpu_id]["cmd"]
                    del running_commands[gpu_id]
                    console.print(f"[INFO] Task on GPU {gpu_id} killed: {killed_cmd}", style="bold red")

                elif gpu_id in default_running:
                    screen_name = f"default_gpu{gpu_id}"
                    console.print(f"[INFO] Killing default screen: {screen_name} (GPU {gpu_id})...", style="bold red")
                    subprocess.run(f"screen -S {screen_name} -X quit", shell=True)
                    del default_running[gpu_id]
                    # ✅ 设置该 GPU 在未来 N 秒内不再自动运行 default
                    delay = 300  # 秒，可改为参数
                    default_block_until[gpu_id] = time.time() + delay
                    console.print(f"[INFO] Default task on GPU {gpu_id} killed. Default will restart after {delay}s.",
                                  style="bold yellow")

                else:
                    console.print(f"[WARN] No task running on GPU {gpu_id}", style="yellow")
            elif command == "update":
                update_task_file(args)

            elif command == "hold":
                if len(parts) == 2 and parts[1].lower() in ["on", "off"]:
                    if parts[1].lower() == "on":
                        enable_default_task.set()
                    else:
                        enable_default_task.clear()
                    console.print(
                        f"[INFO] Default task holding is now {'enabled' if enable_default_task.is_set() else 'disabled'}",
                        style="cyan")
                else:
                    console.print("[ERROR] Usage: hold on|off", style="bold red")

            elif command == "clear-completed":
                completed_commands.clear()
                console.print("[INFO] Completed tasks cleared.", style="dim")

            else:
                console.print(f"[ERROR] Unknown command: {command}", style="bold red")

        except Exception as e:
            console.print(f"[ERROR] Failed to parse input: {e}", style="bold red")

def update_task_file(args):
    new_task_count = 0
    if not os.path.isfile(args.task_file):
        console.print(f"[ERROR] Task file not found: {args.task_file}", style="bold red")
        return

    all_new_commands = set()
    with open(args.task_file, 'r') as f:
        for line in f:
            cmd = line.strip()
            if not cmd or cmd.startswith("#"):
                continue
            if cmd in running_commands.values():
                continue
            all_new_commands.add(cmd)

    # 清空 pending 队列
    with pending_commands.mutex:
        pending_commands.queue.clear()

    # 添加新任务
    for cmd in all_new_commands:
        pending_commands.put({"cmd": cmd})
        loaded_commands_set.add(cmd)
        new_task_count += 1

    # 终止所有默认任务
    if new_task_count > 0:
        for gpu_id in list(default_running.keys()):
            screen_name = f"default_gpu{gpu_id}"
            subprocess.run(f"screen -S {screen_name} -X quit", shell=True)
            console.print(f"[INFO] Default task on GPU {gpu_id} terminated to run new task", style="bold yellow")
            del default_running[gpu_id]

    if new_task_count == 0:
        console.print("[INFO] No new tasks found.", style="dim")
    else:
        console.print(f"[INFO] {new_task_count} new task(s) loaded and applied.", style="green")


def command_to_filename(command: str) -> str:
    # 简化命令做一部分哈希，防止命名冲突或过长
    safe_base = command.replace(" ", "_").replace("/", "_")[:60]
    hash_suffix = hashlib.md5(command.encode()).hexdigest()[:8]
    return f"{safe_base}_{hash_suffix}.txt"

def get_args():
    parser = argparse.ArgumentParser(description="GPU Task Scheduler")

    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0, 1, 2, 3],
        help="List of GPU IDs to monitor and use"
    )
    parser.add_argument(
        "--mem", type=int, default=20,
        help="Minimum free memory (G) required to schedule a task"
    )
    parser.add_argument(
        "--task-file", type=str, default='./tools/task_list.bash',
        help="Path to the task file; each line is a command to run"
    )
    parser.add_argument(
        "--log-dir", type=str, default="/mnt/sdb/lx/result/run_logs/",
        help="Directory to save logs for each task"
    )
    parser.add_argument(
        "--default-cmd", type=str, default="python ./tools/base_train.py",
        help="Command to run as default when task queue is empty and GPU is free"
    )
    # parser.add_argument(
    #     "--default-cmd", type=str, default="python ./tools/auto_run_test.py --epoch 1",
    #     help="Command to run as default when task queue is empty and GPU is free"
    # )
    parser.add_argument(
        "--poll-interval", type=int, default=1,
        help="Polling interval in seconds"
    )
    parser.add_argument(
        "--conda-path", type=str, default="~/anaconda3/etc/profile.d/conda.sh",
        help="Path to conda.sh for environment activation"
    )
    parser.add_argument(
        "--conda-env", type=str, default="mmlab",
        help="Name of the conda environment to activate"
    )
    parser.add_argument(
        "--project-path", type=str, default="/home/b1/lx/mmdetection-main/",
        help="Path to the working project directory"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # 初始化 GPU 列表
    shared_gpus = list(args.gpus)

    # 所有待运行命令（格式：字典）
    pending_commands = queue.Queue()
    running_commands = {}
    completed_commands = []
    # 创建 logs 目录
    os.makedirs(args.log_dir, exist_ok=True)
    # 从文件读取任务命令
    if not os.path.isfile(args.task_file):
        raise FileNotFoundError(f"Task file not found: {args.task_file}")

    update_task_file(args)

    # 启动线程
    threading.Thread(target=monitor_and_run, args=(args,), daemon=True).start()
    threading.Thread(target=check_completed, args=(args,), daemon=True).start()
    threading.Thread(target=gpu_input_listener, daemon=True).start()
    display_status()
