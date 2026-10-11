"""Compatibility command; prefer python -m breakhis train."""
import sys
from breakhis.cli import main as command_main


def main():
    args = sys.argv[1:]
    if '--model' not in args:
        args += ['--model', 'resnet']
    command_main(['train', *args])


if __name__ == '__main__':
    main()
