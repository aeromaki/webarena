from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
from argparse import Namespace

from pathlib import Path
import json
from logging import getLogger


logger = getLogger("logger")


ActionSetTag = Literal["id_accessibility_tree", "playwright"]
ObservationType = Literal["accessibility_tree", "html", "image"]

AgentType = Literal["prompt", "teacher_forcing"]

Provider = Literal["openai", "huggingface"]
Mode = Literal["chat", "generation"]


@dataclass(frozen=True)
class WebArenaConfig:
    """
    dataclass representation of argparse Namespace for type hints
    """
    agent: AgentConfig
    lm: LMConfig
    example: ExampleConfig
    logging: LoggingConfig

    render: bool = False
    slow_mo: int = 0
    action_set_tag: ActionSetTag = "id_accessibility_tree"
    observation_type: ObservationType = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    render_screenshot: bool = True
    save_trace_enabled: bool = False
    sleep_after_execution: float = 2.0
    max_steps: int = 30

    @staticmethod
    def from_args(args: Namespace) -> WebArenaConfig:
        return WebArenaConfig(
            AgentConfig.from_args(args),
            LMConfig.from_args(args),
            ExampleConfig.from_args(args),
            LoggingConfig.from_args(args),
            args.render,
            args.slow_mo,
            args.action_set_tag,
            args.observation_type,
            args.current_viewport_only,
            args.viewport_width,
            args.viewport_height,
            args.save_trace_enabled,
            args.sleep_after_execution,
            args.max_steps
        )

    @staticmethod
    def from_json(path: str) -> WebArenaConfig:
        with open(path, "r") as f:
            config_dict = json.load(f)
        config_dict["agent"] = AgentConfig(**config_dict["agent"])
        config_dict["lm"] = LMConfig(**config_dict["lm"])
        config_dict["example"] = ExampleConfig(**config_dict["example"])
        config_dict["logging"] = LoggingConfig(**config_dict["logging"])
        return WebArenaConfig(**config_dict)

    def dump(self) -> None:
        """
        originally `dump_config` in run.py
        save config as json in self.logging.result_dir
        """
        config_file_path = Path(self.logging.result_dir) / "config.json"

        config_dict = asdict(self)
        config_dict["agent"] = asdict(self.agent)
        config_dict["lm"] = asdict(self.lm)
        config_dict["example"] = asdict(self.example)
        config_dict["logging"] = asdict(self.logging)

        if not config_file_path.exists():
            with open(config_file_path, "w") as f:
                json.dump(config_dict, f, indent=4)
                logger.info(f"Dump config to {config_file_path}")

@dataclass(frozen=True)
class AgentConfig:
    """
    config about agent
    """
    agent_type: AgentType = "prompt"
    instruction_path: str = "agents/prompts/state_action_agent.json"
    parsing_failure_th: int = 3
    repeating_action_failure_th: int = 3

    @staticmethod
    def from_args(args: Namespace) -> AgentConfig:
        return AgentConfig(
            args.agent_type,
            args.instruction_path,
            args.parsing_failure_th,
            args.repeating_action_failure_th
        )

@dataclass(frozen=True)
class LMConfig:
    """
    config about LM
    """
    provider: Provider = "openai"
    model: str = "gpt-3.5-turbo-0613"
    mode: Mode = "chat"
    temperature: float = 1.0
    top_p: float = 0.9
    context_length: int = 0
    max_tokens: int = 384
    stop_token: str | None = None
    max_retry: int = 1
    max_obs_length: int = 1920
    model_endpoint: str = ""

    @staticmethod
    def from_args(args: Namespace) -> LMConfig:
        return LMConfig(
            args.provider,
            args.model,
            args.mode,
            args.temperature,
            args.top_p,
            args.context_length,
            args.max_tokens,
            args.stop_token,
            args.max_retry,
            args.max_obs_length,
            args.model_endpoint
        )

@dataclass(frozen=True)
class ExampleConfig:
    """
    config about example
    """
    test_start_idx: int = 0
    test_end_idx: int = 1000

    @staticmethod
    def from_args(args: Namespace) -> ExampleConfig:
        return ExampleConfig(
            args.test_start_idx,
            args.test_end_idx
        )

@dataclass(frozen=True)
class LoggingConfig:
    """
    config about logging
    """
    result_dir: str = ""

    @staticmethod
    def from_args(args: Namespace) -> LoggingConfig:
        return LoggingConfig(
            args.result_dir
        )