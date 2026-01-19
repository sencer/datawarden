import sys

from loguru import logger

# Global logger configuration for benchmarks to strip timestamps/metadata
logger.remove()
logger.add(sys.stderr, format="<level>{message}</level>")
