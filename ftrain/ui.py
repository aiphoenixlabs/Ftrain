import sys, time, threading
from typing import Dict, Any, Optional

CLEAR_LINE = "\033[K"
RESET = "\033[0m"

def fire_header():
    sep = "=" * 60
    art = r"""
 ███████╗████████╗██████╗  █████╗ ██╗███╗   ██╗
 ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
 █████╗     ██║   ██████╔╝███████║██║██╔██╗ ██║
 ██╔══╝     ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
 ██║        ██║   ██║  ██║██║  ██║██║██║ ╚████║
 ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
    """
    print(f"\n\033[38;5;208m{art}{RESET}\n\033[38;5;239m{sep}{RESET}")
    print(f"\033[1;38;5;214m{'🔥' * 8}  FIRE TRAIN ENGINE v1.0.0  {'🔥' * 8}{RESET}")
    print(f"\033[38;5;239m{sep}{RESET}\n")

def gradient_bar(progress: float, width: int = 20, from_blue_to_orange: bool = False) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    bar = ""
    if from_blue_to_orange:
        if progress <= 0.33:
            fill_code = "\033[48;5;39m"
        elif progress <= 0.66:
            fill_code = "\033[48;5;214m"
        else:
            fill_code = "\033[48;5;196m"
    else:
        if progress <= 0.25:
            fill_code = "\033[48;5;19m"
        elif progress <= 0.50:
            fill_code = "\033[48;5;45m"
        elif progress <= 0.75:
            fill_code = "\033[48;5;226m"
        else:
            fill_code = "\033[48;5;208m"
    empty_code = "\033[48;5;236m"
    for _ in range(filled):
        bar += f"{fill_code} {RESET}"
    for _ in range(filled, width):
        bar += f"{empty_code} {RESET}"
    return f"|{bar}|"

def print_train_table(step, total_steps, loss, val_loss, lr, grad_norm, captain_msg=""):
    progress = step / total_steps if total_steps > 0 else 0
    bar = gradient_bar(progress, 15)
    loss_str = f"{loss:.4f}" if loss is not None else " N/A "
    val_str = f"{val_loss:.4f}" if val_loss is not None else " N/A "
    lr_str = f"{lr:.2e}" if lr is not None else " N/A "
    grad_str = f"{grad_norm:.4f}" if grad_norm is not None else " N/A "
    cap_str = f" | 🧠 \033[38;5;51m{captain_msg}{RESET}" if captain_msg else ""
    sys.stdout.write(
        f"\r{CLEAR_LINE}📊 Step \033[1m{step:>4}/{total_steps:<4}{RESET} | "
        f"Loss: \033[38;5;208m{loss_str}{RESET} | "
        f"Val: \033[38;5;45m{val_str}{RESET} | "
        f"LR: {lr_str} | Grad: {grad_str} | {bar}{cap_str}\n"
    )
    sys.stdout.flush()

def print_merge_progress(current, total, message=""):
    progress = current / total if total > 0 else 0
    bar = gradient_bar(progress, 25, from_blue_to_orange=True)
    msg = f" \033[38;5;245m({message}){RESET}" if message else ""
    sys.stdout.write(f"\r{CLEAR_LINE}🧩 Merge [{bar}] \033[1m{progress*100:>3.0f}%\033[0m{msg} ")
    sys.stdout.flush()
    if progress >= 1.0:
        sys.stdout.write("\n")
        sys.stdout.flush()

def print_captain_report(report: str):
    width = 60
    print(f"\n\033[38;5;239m{'=' * width}{RESET}")
    print(f"\033[1;38;5;51m{'🧠 CAPTAIN ANALYSIS REPORT'.center(width)}{RESET}")
    print(f"\033[38;5;239m{'-' * width}{RESET}")
    for line in report.split("\n"):
        print(f" {line}")
    print(f"\033[38;5;239m{'=' * width}{RESET}\n")

def print_final_summary(stats: Dict[str, Any]):
    width = 55
    print(f"\n\033[38;5;239m{'=' * width}{RESET}")
    print(f"\033[1;38;5;46m{'✅ PROCESS COMPLETED SUCCESSFULLY'.center(width)}{RESET}")
    print(f"\033[38;5;239m{'=' * width}{RESET}")
    for k, v in stats.items():
        key_str = str(k).replace('_', ' ').title()
        val_str = str(v)
        if "http" in val_str:
            val_str = f"\033[4;38;5;45m{val_str}{RESET}"
        print(f" \033[1m{key_str:<25}\033[0m : \033[38;5;45m{val_str}{RESET}")
    print(f"\033[38;5;239m{'=' * width}{RESET}\n")

class LoadingBar:
    def __init__(self, message="Loading model", real_progress=True):
        self.message = message
        self.real = real_progress
        self.stop_event = threading.Event()
        self._progress = 0.0
        self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.done()

    def start(self):
        if self.real:
            sys.stdout.write(f"{CLEAR_LINE}📦 {self.message} (0%)")
            sys.stdout.flush()
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        i = 0
        while not self.stop_event.is_set():
            i = (i + 1) % 101
            bar = gradient_bar(i/100, 20, from_blue_to_orange=True)
            sys.stdout.write(f"\r{CLEAR_LINE}📦 {self.message} [{bar}] {i}%")
            sys.stdout.flush()
            time.sleep(0.05)

    def update(self, current, total):
        self._progress = current / total if total > 0 else 0
        bar = gradient_bar(self._progress, 20, from_blue_to_orange=True)
        sys.stdout.write(f"\r{CLEAR_LINE}📦 {self.message} [{bar}] {self._progress*100:.0f}%")
        sys.stdout.flush()

    def done(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        bar = gradient_bar(1.0, 20, from_blue_to_orange=True)
        sys.stdout.write(f"\r{CLEAR_LINE}📦 {self.message} [{bar}] 100%\n")
        sys.stdout.flush()
