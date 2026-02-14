import logging
import os


def configure_logging(study_area_dir, console_level=logging.INFO):
    """Configure logging for the connectome package.

    - Console: console_level and above
    - File: DEBUG and above -> {study_area_dir}/run.log (append mode)
    """
    root = logging.getLogger("connectome")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    os.makedirs(study_area_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(study_area_dir, "run.log"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
