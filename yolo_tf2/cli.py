import argparse
import sys

from yolo_tf2.config.cli_args import DETECTION, EVALUATION, GENERAL, TRAINING
from yolo_tf2.utils.cli_utils import (add_args, detect, display_commands,
                                      display_section, evaluate, train)


def execute():
    """
    Train or evaluate or detect based on cli args.
    Returns:
        None
    """
    valid_commands = {
        'train': ('TRAINING', TRAINING, train),
        'evaluate': ('EVALUATION', EVALUATION, evaluate),
        'detect': ('DETECTION', DETECTION, detect),
    }
    if len(sys.argv) == 1:
        display_commands()
        return

    cli_args = sys.argv
    total = len(cli_args)
    command = cli_args[1]
    help_flags = any(('-h' in cli_args, '--help' in cli_args))

    if command in valid_commands and total == 2:
        display_section(valid_commands[command][0])
        return
    if help_flags and total == 2:
        display_commands(True)
        return
    if total == 3 and command in valid_commands and help_flags:
        display_section(valid_commands[command][0])
        return
    if command not in valid_commands:
        print(f'Invalid command {command}')
        return
    parser = argparse.ArgumentParser()
    del sys.argv[1]
    parser = add_args(GENERAL, parser)
    valid_commands[command][2](parser)


if __name__ == '__main__':
    execute()
