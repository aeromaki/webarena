"""Script to run end-to-end evaluation on the benchmark"""
import logging
import os
import random
import time
from pathlib import Path

import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent_from_config
)
from agent.prompts import *
from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)

from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router


from inference.config import WebArenaConfig, get_config
from inference.utils import (
    prepare,
    create_test_file_list,
    log_error_file
)
from inference.pipes import (
    create_env_from_config,
    get_intent_and_task_id,
    get_next_action
)

from typing import Optional




LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def test(
    config: WebArenaConfig,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    result_dir = config.logging.result_dir

    scores = []

    env = create_env_from_config(config)

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, result_dir, config.action_set_tag
            )

            # get intent
            intent, task_id = get_intent_and_task_id(config_file)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            # reset
            agent.reset(config_file)
            obs, info = env.reset(options={"config_file": config_file})

            # init state_info, trajectory
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory: Trajectory = [state_info]

            # init metadata
            meta_data = {"action_history": ["None"]}

            while True:
                # get new action
                action = get_next_action(
                    config,
                    trajectory,
                    agent,
                    intent,
                    meta_data
                )

                # update trajectory
                trajectory.append(action)

                # get description about new action
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=config.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )

                render_helper.render(
                    action, state_info, meta_data, config.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                # get new env state
                obs, _, terminated, _, info = env.step(action)

                # update state_info, trajectory
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action placeholder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if config.save_trace_enabled:
                env.save_trace(
                    Path(result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            log_error_file(result_dir, config_file, e)

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")




if __name__ == "__main__":
    config = get_config()
    prepare(config.logging.result_dir, LOG_FILE_NAME)

    test_file_list = create_test_file_list(
        config.example.test_start_idx,
        config.example.test_end_idx,
        config.logging.result_dir
    )

    if len(test_file_list) > 0:
        logger.info(f"Total {len(test_file_list)} tasks left")
        config.dump()
        agent = construct_agent_from_config(config)
        test(config, agent, test_file_list)

    else:
        logger.info("No task left to run")
