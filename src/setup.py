"""Docstring for v25.06.00.Techical project.setup

File containing utility functions for loading tube parameters and plugin lists from JSON files,
as well as functions for colored terminal output."""
import os
import json

def load_tube_parameters(path):
    """Carica i parametri dei tubi da tube_parameters.json."""
    json_file = os.path.join(path, "tube_parameters.json")
    try:
        with open(json_file, "r") as f: 
            return json.load(f)
    except FileNotFoundError:
        print(f"\033[1;91m[plot] Error: {json_file} not found.\033[0m")
        exit()
    
def load_plugin_list(path):
    """Carica lista di plugins da plugin_list.json"""
    json_file = os.path.join(path, "plugin_list.json")
    try:
        with open(json_file, "r") as f:
            plugin_list = json.load(f)
            return plugin_list["plugins"]
    except FileNotFoundError:
        print(f"\033[1;91m[plot] Error: {json_file} not found.\033[0m")
        exit()

# Colori ANSI
ANSI_COLORS = {
    "red":    "\033[1;91m",
    "green":  "\033[1;92m",
    "yellow": "\033[1;93m",
    "blue":   "\033[1;94m",
    "magenta":"\033[1;95m",
    "cyan":   "\033[1;96m",
    "white":  "\033[1;97m",
    "reset":  "\033[0;0m"
}

def colored(text, color):
    """Ritorna una stringa colorata con ANSI escape codes."""
    return f"{ANSI_COLORS.get(color,'white')}{text}{ANSI_COLORS['reset']}"

def colored_tube_number(tube):
    """Restituisce il numero del tubo con il suo colore ANSI."""
    mapping = {1: "red", 2: "green", 3: "blue"}
    return colored(str(tube), mapping.get(tube, "white"))


