"""
CLI Utilities Module
Enhanced formatting and display utilities for beautiful CLI output
"""

from colorama import Fore, Back, Style, init
from tabulate import tabulate
from typing import Optional, List, Dict
import sys

# Initialize colorama for Windows support
init(autoreset=True)

# Windows consoles often default to cp1252; emoji/box-drawing in output then raises UnicodeEncodeError.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


class Colors:
    """Color constants for CLI output"""
    HEADER = Fore.CYAN + Style.BRIGHT
    SUCCESS = Fore.GREEN + Style.BRIGHT
    WARNING = Fore.YELLOW + Style.BRIGHT
    ERROR = Fore.RED + Style.BRIGHT
    INFO = Fore.BLUE
    HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
    DIM = Style.DIM
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT


class CLIFormatter:
    """Enhanced CLI formatting utilities"""
    
    @staticmethod
    def print_header(text: str, width: int = 70):
        """Print a fancy header"""
        print("\n" + Colors.HEADER + "═" * width)
        print(f"  {text}")
        print("═" * width + Colors.RESET)
    
    @staticmethod
    def print_subheader(text: str, width: int = 70):
        """Print a subheader"""
        print("\n" + Colors.INFO + "─" * width)
        print(f"  {text}")
        print("─" * width + Colors.RESET)
    
    @staticmethod
    def print_success(text: str, icon: str = "✓"):
        """Print success message"""
        print(Colors.SUCCESS + f"{icon} {text}" + Colors.RESET)
    
    @staticmethod
    def print_warning(text: str, icon: str = "⚠"):
        """Print warning message"""
        print(Colors.WARNING + f"{icon} {text}" + Colors.RESET)
    
    @staticmethod
    def print_error(text: str, icon: str = "✗"):
        """Print error message"""
        print(Colors.ERROR + f"{icon} {text}" + Colors.RESET)
    
    @staticmethod
    def print_info(text: str, icon: str = "ℹ"):
        """Print info message"""
        print(Colors.INFO + f"{icon} {text}" + Colors.RESET)
    
    @staticmethod
    def print_step(step_num: int, total_steps: int, text: str):
        """Print a step indicator"""
        print(f"\n{Colors.HIGHLIGHT}[STEP {step_num}/{total_steps}]{Colors.RESET} {Colors.BOLD}{text}{Colors.RESET}")
    
    @staticmethod
    def print_box(title: str, content: list, width: int = 70):
        """Print content in a box"""
        print("\n" + Colors.INFO + "┌" + "─" * (width - 2) + "┐")
        print(f"│ {Colors.BOLD}{title}{Colors.RESET}{Colors.INFO}" + " " * (width - len(title) - 3) + "│")
        print("├" + "─" * (width - 2) + "┤")
        
        for line in content:
            # Handle long lines
            if len(line) > width - 4:
                line = line[:width - 7] + "..."
            padding = width - len(line) - 4
            print(f"│ {Colors.RESET}{line}{Colors.INFO}" + " " * padding + "│")
        
        print("└" + "─" * (width - 2) + "┘" + Colors.RESET)
    
    @staticmethod
    def print_table(headers: List, rows: List, highlight_row: Optional[int] = None):
        """Print a formatted table"""
        # Add color to headers
        colored_headers = [Colors.BOLD + str(h) + Colors.RESET for h in headers]
        
        # Highlight specific row if requested
        if highlight_row is not None and 0 <= highlight_row < len(rows):
            rows[highlight_row] = [
                Colors.SUCCESS + "★ " + str(cell) + Colors.RESET 
                if i == 0 else Colors.SUCCESS + str(cell) + Colors.RESET
                for i, cell in enumerate(rows[highlight_row])
            ]
        
        table = tabulate(rows, headers=colored_headers, tablefmt="simple", floatfmt=".4f")
        print("\n" + table)
    
    @staticmethod
    def print_metric_card(title: str, metrics: dict):
        """Print a card showing metrics"""
        print(f"\n{Colors.HIGHLIGHT}┌─ {title} " + "─" * (50 - len(title)) + "┐" + Colors.RESET)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            key_display = key.replace('_', ' ').title()
            padding = 40 - len(key_display)
            print(f"{Colors.INFO}│{Colors.RESET} {key_display}:" + " " * padding + f"{Colors.BOLD}{formatted_value}{Colors.RESET}")
        
        print(Colors.HIGHLIGHT + "└" + "─" * 50 + "┘" + Colors.RESET)
    
    @staticmethod
    def print_progress_bar(current: int, total: int, prefix: str = "", length: int = 40):
        """Print a simple progress bar"""
        percent = current / total
        filled = int(length * percent)
        bar = "█" * filled + "░" * (length - filled)
        
        print(f"\r{prefix} {Colors.INFO}[{bar}]{Colors.RESET} {percent*100:.0f}%", end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    @staticmethod
    def get_input(prompt: str, default: Optional[str] = None, color=Colors.HIGHLIGHT) -> str:
        """Get colored input from user"""
        if default:
            prompt_text = f"{color}{prompt} (default: {default}): {Colors.RESET}"
        else:
            prompt_text = f"{color}{prompt}: {Colors.RESET}"
        
        return input(prompt_text).strip()
    
    @staticmethod
    def print_summary_box(items: dict):
        """Print a summary box with key-value pairs"""
        print(f"\n{Colors.SUCCESS}╔════════════════════════════════════════════════════════════════╗")
        print(f"║  {Colors.BOLD}SUMMARY{Colors.RESET}{Colors.SUCCESS}                                                       ║")
        print(f"╠════════════════════════════════════════════════════════════════╣{Colors.RESET}")
        
        for key, value in items.items():
            key_str = str(key)
            value_str = str(value)
            padding = 60 - len(key_str) - len(value_str)
            print(f"{Colors.SUCCESS}║{Colors.RESET}  {Colors.BOLD}{key_str}:{Colors.RESET} " + " " * padding + f"{value_str}  {Colors.SUCCESS}║{Colors.RESET}")
        
        print(f"{Colors.SUCCESS}╚════════════════════════════════════════════════════════════════╝{Colors.RESET}")


class ProgressTracker:
    """Track and display progress for operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
    
    def update(self, step_name: str = ""):
        """Update progress"""
        self.current_step += 1
        CLIFormatter.print_progress_bar(
            self.current_step, 
            self.total_steps,
            prefix=f"{self.description}: {step_name}"
        )
    
    def complete(self):
        """Mark as complete"""
        CLIFormatter.print_success(f"{self.description} completed!")


def clear_screen():
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print application banner"""
    banner = f"""
{Colors.HEADER}
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        🤖 DATASET-DRIVEN AUTOMATED ML CLI TOOL 🤖              ║
    ║                                                                  ║
    ║              Intelligent • Explainable • Efficient               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
{Colors.RESET}
    """
    print(banner)
