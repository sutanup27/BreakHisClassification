"""Compatibility command; prefer python -m breakhis evaluate."""
import sys
from breakhis.cli import main as command_main


def main():
    args = sys.argv[1:]
    command_main(['evaluate', *args])


if __name__ == '__main__':
    main()
