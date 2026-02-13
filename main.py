"""The main module of the ttt4145-semester-project package."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Print a welcome message and logs the start of the main function."""
    logger = logging.getLogger(__name__)
    logger.info("Starting ttt4145-semester-project main function.")


if __name__ == "__main__":
    main()
