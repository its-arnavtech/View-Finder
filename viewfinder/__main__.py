"""Allow ``python -m viewfinder`` to behave like the installed CLI."""

from .cli import main


if __name__ == "__main__":
    main()
