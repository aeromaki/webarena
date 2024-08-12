import json
import os
import tempfile
import subprocess
from typing import Any

from agent import Agent
from browser_env import (
    ScriptBrowserEnv,
    Action,
    Trajectory,
    create_stop_action
)
from browser_env.auto_login import get_site_comb_from_filepath

from inference.config import WebArenaConfig
from .early_stop import early_stop


def create_env_from_config(config: WebArenaConfig) -> ScriptBrowserEnv:
    env = ScriptBrowserEnv(
        headless=not config.render,
        slow_mo=config.slow_mo,
        observation_type=config.observation_type,
        current_viewport_only=config.current_viewport_only,
        viewport_size={
            "width": config.viewport_width,
            "height": config.viewport_height,
        },
        save_trace_enabled=config.save_trace_enabled,
        sleep_after_execution=config.sleep_after_execution,
    )
    return env


def get_intent_and_task_id(config_file: str) -> tuple[str, str]:
    with open(config_file) as f:
        _c = json.load(f)
        intent: str = _c["intent"]
        task_id = _c["task_id"]
        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()
            # subprocess to renew the cookie
            subprocess.run(
                [
                    "python",
                    "browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ]
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(config_file, "w") as f:
                json.dump(_c, f)
    return intent, task_id


def get_next_action(
    config: WebArenaConfig,
    trajectory: Trajectory,
    agent: Agent,
    intent: str,
    meta_data: dict[str, Any]
) -> Action:
    """
    Gets next action of agent
    Creates stop action if early stop or error occurs
    """
    action: Action
    # early stop
    if (stop_info := early_stop(
        trajectory,
        config.max_steps,
        config.agent.parsing_failure_th,
        config.agent.repeating_action_failure_th
    )) is not None:
        action = create_stop_action(f"Early stop: {stop_info}")

    else:
        try:
            action = agent.next_action(
                trajectory, intent, meta_data=meta_data
            )
        except ValueError as e:
            action = create_stop_action(f"ERROR: {str(e)}")

    return action