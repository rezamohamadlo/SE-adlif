"""Compatibility entry point; use runner.py --models MODEL for new commands."""
import sys

from runner import main


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if '--models' not in arguments:
        arguments = ['--models', *arguments]
    main(arguments)
