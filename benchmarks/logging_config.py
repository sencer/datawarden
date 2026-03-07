import sys

from loguru import logger


def configure_logger() -> None:
  # Global logger configuration for benchmarks to strip timestamps/metadata
  logger.remove()
  logger.add(sys.stderr, format="<level>{message}</level>")


configure_logger()
