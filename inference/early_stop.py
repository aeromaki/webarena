from typing import Optional

from browser_env import Action, ActionTypes, Trajectory
from browser_env.actions import is_equivalent


def early_stop(
    trajectory: Trajectory,
    max_steps: int,
    parsing_failure_th: int,
    repeating_action_failure_th: int,
) -> Optional[str]:
    """
    Check whether need to early stop
    Returns None if false, str if true
    """
    if (msg := _max_step(trajectory, max_steps)) is not None:
        return msg
    elif (msg := _parsing_failure(trajectory, parsing_failure_th)) is not None:
        return msg
    elif (msg := _repeating_action_failure(trajectory, repeating_action_failure_th)) is not None:
        return msg
    else:
        return None


def _max_step(trajectory: Trajectory, max_steps: int) -> Optional[str]:
    """
    Case: reach the max step
    """
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return f"Reach max steps {max_steps}"
    else:
        return None

def _parsing_failure(trajectory: Trajectory, threshold: int) -> Optional[str]:
    """
    Case: parsing failure for k times
    """
    k = threshold
    last_k_actions: list[Action] = trajectory[1::2][-k:] # type: ignore
    if len(last_k_actions) >= k\
        and all(
        [
            action["action_type"] == ActionTypes.NONE
            for action in last_k_actions
        ]
    ):
        return f"Failed to parse actions for {k} times"
    else:
        return None

def _repeating_action_failure(trajectory: Trajectory, threshold: int) -> Optional[str]:
    """
    Case: same action for k times
    """
    k = threshold
    last_k_actions: list[Action] = trajectory[1::2][-k:] # type: ignore
    action_seq: list[Action] = trajectory[1::2] # type: ignore

    if len(action_seq) == 0:
        return None

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k\
            and all(
            [
                is_equivalent(action, last_action)
                for action in last_k_actions
            ]
        ):
            return f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return f"Same typing action for {k} times"

    return None