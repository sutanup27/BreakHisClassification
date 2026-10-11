"""Compatibility command; prefer python -m breakhis prune."""
import sys
from breakhis.cli import main as command_main


def main():
    args = sys.argv[1:]
    command_main(['prune', *args])


if __name__ == '__main__':
    main()
