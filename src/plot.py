"""
Docstring for v25.06.00.Techical project.plot2

Script to plot 3D workspace CSV files using matplotlib.
Supports single and multiple file plotting with flexible file selection.
"""
import argparse
import logging
import math
import os
import sys
import glob

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

from PIDController import WORKSPACE_PATH

# Simple colored-print helpers (use these when you want colored console output
# without touching the logging configuration). Uses ANSI escape codes; terminals
# that don't support them will show the raw escapes.
RESET = "\033[0m"
RED_BOLD = "\033[1;91m"
YELLOW_BOLD = "\033[1;93m"
GREEN_BOLD = "\033[1;92m"

def print_error(msg):
    """Print an error message in bold red (keeps signature simple)."""
    print(f"{RED_BOLD}{msg}{RESET}")


def print_warning(msg):
    """Print a warning message in bold yellow."""
    print(f"{YELLOW_BOLD}{msg}{RESET}")


def print_info(msg):
    """Print an informational message in bold green."""
    print(f"{GREEN_BOLD}{msg}{RESET}")

def load_workspace(file_path):
    """Load a CSV workspace file. Try to auto-detect delimiter.

    Returns a pandas.DataFrame or None on failure.
    """
    if not os.path.exists(file_path):
        print_error(f"[plot] File not found: {file_path}")
        return None

    # Let pandas try to detect the separator (engine='python' supports sep=None autodetect)
    try:
        data = pd.read_csv(file_path, sep=None, engine="python")
        print_info(f"[plot] Loaded {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print_error(f"[plot] Failed to read {file_path}: {e}")
        return None


def plot_single_workspace(data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    try:
        c = data['Ins3'] if 'Ins3' in data.columns else None
        scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=c, cmap='jet', s=20)
    except KeyError as e:
        print_error(f"Missing expected column in data: {e}")
        return

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    if 'Ins3' in data.columns:
        fig.colorbar(scatter, ax=ax)
    plt.show()


def plot_multiple(files):
    """Plot multiple workspaces. `files` is an ordered dict-like mapping label->path."""
    n = len(files)
    if n == 0:
        print_warning("No files to plot")
        return

    # Layout: up to 3 columns per row
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    for i, (label, path) in enumerate(files.items(), start=1):
        data = load_workspace(path)
        if data is None:
            continue

        ax = fig.add_subplot(rows, cols, i, projection='3d')
        try:
            c = data['Ins3'] if 'Ins3' in data.columns else None
            scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=c, cmap='jet', s=20)
        except KeyError as e:
            print_error(f"Missing expected column in {path}: {e}")
            continue

        ax.set_title(label)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)

        # Limiti degli assi
        ax.set_xlim([-20, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([0, 240])

    plt.tight_layout()
    plt.show()


def get_last_workspace():
    files = glob.glob(os.path.join(WORKSPACE_PATH, "workspace_*.csv"))
    if not files:
        # Use the colored print helper so we don't need to change the logger.
        print_error("[plot] No workspace files found.")
        sys.exit(1)

    latest = max(files, key=os.path.getmtime)
    return latest


def find_files_with_suffix(suffix):
    """Return a mapping label->path for files that end with _{suffix}.csv in WORKSPACE_PATH."""
    pattern = os.path.join(WORKSPACE_PATH, f"*_{suffix}.csv")
    matches = sorted(glob.glob(pattern))
    files = {}
    for p in matches:
        label = os.path.splitext(os.path.basename(p))[0]
        files[label] = p
    return files


def find_specific_file_with_suffix(key, suffix):
    """Return a path for a file that contains `key` in its basename and ends with _{suffix}.csv.

    If multiple matches exist the first (sorted) one is returned. Returns None if none found.
    The `key` is a short identifier like 'nominal', 'pos' or 'neg' and is matched against the
    filename (without extension).
    """
    pattern = os.path.join(WORKSPACE_PATH, f"*_{suffix}.csv")
    matches = sorted(glob.glob(pattern))
    for p in matches:
        name = os.path.splitext(os.path.basename(p))[0]
        if key in name:
            return p
    return None


def build_default_files():
    return {
        "nominal": os.path.join(WORKSPACE_PATH, "workspace_nominal.csv"),
        "neg": os.path.join(WORKSPACE_PATH, "workspace_neg_var.csv"),
        "pos": os.path.join(WORKSPACE_PATH, "workspace_pos_var.csv"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot workspace CSV files (3D)")
    parser.add_argument('mode', nargs='?', default='help',
                        help="Mode: nominal, neg, pos, last, all or 'all <suffix>' to load files *_<suffix>.csv")
    parser.add_argument('suffix', nargs='?', help="Suffix number to use with mode 'all' (e.g. 12 to match *_12.csv)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='[plot] %(levelname)s: %(message)s')

    FILES = build_default_files()

    mode = args.mode.lower()

    if mode == 'last':
        file_path = get_last_workspace()
        data = load_workspace(file_path)
        if data is not None:
            plot_single_workspace(data, "3D Workspace (last)")

    elif mode in FILES:
        # Allow specifying a suffix for specific modes (pos/neg/nominal).
        path = FILES[mode]
        if args.suffix:
            specific = find_specific_file_with_suffix(mode, args.suffix)
            if specific:
                path = specific
                # logging.info("Using %s for mode %s (suffix %s)", os.path.basename(path), mode, args.suffix)
                print_info(f"Using {os.path.basename(path)} for mode {mode} (suffix {args.suffix})")
            else:
                print_warning(f"No {mode}-specific file found with suffix {args.suffix}; falling back to default {os.path.basename(path)}")

        data = load_workspace(path)
        if data is not None:
            plot_single_workspace(data, f"3D Workspace {mode}")

    elif mode == 'all':
        # If suffix provided, load files matching *_<suffix>.csv but always include the
        # nominal workspace in the plotting set. If no matching suffixed files are
        # found, we still plot the nominal workspace and emit a warning instead of
        # exiting.
        if args.suffix:
            matched = find_files_with_suffix(args.suffix)
            # Start with nominal workspace first
            files = {"nominal": FILES.get("nominal")}

            if matched:
                # Merge matched files while avoiding duplicate nominal entry
                for k, v in matched.items():
                    if v != FILES.get("nominal"):
                        files[k] = v
            else:
                print_warning(f"No files found matching suffix: {args.suffix}. Plotting only nominal.")

            plot_multiple(files)
        else:
            plot_multiple(FILES)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
