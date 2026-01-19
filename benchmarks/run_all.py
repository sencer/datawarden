import subprocess  # noqa: S404
import sys

from loguru import logger

SCRIPTS = [
  "benchmarks.core",
  "benchmarks.numba_bench",
  "benchmarks.scaling",
  "benchmarks.comprehensive",
  "benchmarks.memory",
]


def run_all() -> None:
  # Configure logger
  logger.remove()
  logger.add(sys.stderr, format="{message}")

  logger.info("=" * 60)
  logger.info("STARTING FULL BENCHMARK SUITE")
  logger.info("=" * 60)

  for module_name in SCRIPTS:
    logger.info(f"\n>>> RUNNING {module_name}")
    try:
      sys.stdout.flush()
      sys.stderr.flush()
      subprocess.run([sys.executable, "-m", module_name], check=True)
    except subprocess.CalledProcessError:
      logger.error(f"FAILED: {module_name}")
      sys.exit(1)

  logger.success("\nALL BENCHMARKS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
  run_all()
