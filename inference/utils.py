from pathlib import Path
import os
import glob
from logging import getLogger

from .config import WebArenaConfig


logger = getLogger("logger")


def prepare(result_dir: str, LOG_FILE_NAME: str) -> None:
    """
    originally `prepare(args: Namespace)` in run.py
    1. Converts all python files in agent/prompts to json files in agent/prompts/jsons
    2. Prepares result dir
    3. Logs the log file
    """
    _prepare_prompt_json()
    _prepare_result_dir(result_dir)
    _log_file(result_dir, LOG_FILE_NAME)


def create_test_file_list(st_idx: int, ed_idx: int, result_dir: str) -> list[str]:
    """
    creates list of task config file path
    """
    test_file_list = []

    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in result_dir:
        test_file_list = _get_unfinished(test_file_list, result_dir)

    return test_file_list


def log_error_file(result_dir: str, config_file: str, exception: Exception) -> None:
    import traceback
    with open(Path(result_dir) / "error.txt", "a") as f:
        f.write(f"[Config file]: {config_file}\n")
        f.write(f"[Unhandled Error] {repr(exception)}\n")
        f.write(traceback.format_exc())  # write stack trace to file


def _prepare_prompt_json() -> None:
    from agent.prompts import to_json
    to_json.run()

def _prepare_result_dir(result_dir: str) -> None:
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

def _log_file(result_dir: str, LOG_FILE_NAME: str) -> None:
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def _get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    """
    originally `get_unfinished` in run.py
    """
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs