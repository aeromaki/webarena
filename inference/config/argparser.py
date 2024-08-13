from argparse import Namespace, ArgumentParser
import time

from .config import WebArenaConfig


def get_config() -> WebArenaConfig:
    """
    parses args and returns WebArenaConfig
    """
    args = _get_args()
    config = WebArenaConfig.from_args(args)
    return config


def _get_args() -> Namespace:
    """
    originally `config()` in run.py
    """
    parser = ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser = _add_webarena_config(parser)
    parser = _add_lm_config(parser)
    parser = _add_example_config(parser)
    parser = _add_logging_config(parser)

    # preset
    parser.add_argument("-p", "--preset", type=str, help="config json")

    args = parser.parse_args()
    _check_action_space(args)
    return args

def _add_webarena_config(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_false",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    return parser

def _add_lm_config(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )
    return parser

def _add_example_config(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)
    return parser

def _add_logging_config(parser: ArgumentParser) -> ArgumentParser:
    result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"

    parser.add_argument("--result_dir", type=str, default=result_dir)
    return parser

def _check_action_space(args: Namespace) -> None:
    """
    check the whether the action space is compatible with the observation space
    """
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )