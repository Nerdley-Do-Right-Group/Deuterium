"""Allows running `python -m deuterium` directly."""

import sys

from deuterium.cli import main

if __name__ == "__main__":
    sys.exit(main())
