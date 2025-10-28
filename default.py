"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

try:
    from contrastive.wandb_logger import WandbLogger

    WANDB_LOGGER_AVAILABLE = True
except ImportError:
    WANDB_LOGGER_AVAILABLE = False


def make_default_logger(
    label: str,
    save_data: bool = True,
    save_dir: str = "logs",
    add_uid: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = "steps",
) -> base.Logger:
    """Makes a default Acme logger.

    Args:
      label: Name to give to the logger.
      save_data: Whether to persist data.
      time_delta: Time (in seconds) between logging events.
      asynchronous: Whether the write function should block or not.
      print_fn: How to print to terminal (defaults to print).
      serialize_fn: An optional function to apply to the write inputs before
        passing them to the various loggers.
      steps_key: Ignored.

    Returns:
      A logger object that responds to logger.write(some_dict).
    """
    del steps_key
    if not print_fn:
        print_fn = logging.info
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]

    if save_data:
        loggers.append(
            csv.CSVLogger(label=label, directory_or_file=save_dir, add_uid=add_uid)
        )

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)
    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, time_delta)

    return logger


def make_wandb_logger(
    label: str,
    save_data: bool = True,
    save_dir: str = "logs",
    add_uid: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = "steps",
    use_wandb: bool = False,
    wandb_project: str = "contrastive-rl",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    wandb_notes: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_config: Optional[dict] = None,
    init_wandb: bool = False,
) -> base.Logger:
    """Makes a logger with optional Wandb support.

    This function creates a logger that can write to terminal, CSV files,
    and optionally to Weights & Biases.

    Args:
      label: Name to give to the logger.
      save_data: Whether to persist data to CSV.
      save_dir: Directory to save CSV logs.
      add_uid: Whether to add unique ID to log directory.
      time_delta: Time (in seconds) between logging events.
      asynchronous: Whether the write function should block or not.
      print_fn: How to print to terminal (defaults to print).
      serialize_fn: An optional function to apply to the write inputs before
        passing them to the various loggers.
      steps_key: Key for step count in logs.
      use_wandb: Whether to enable Wandb logging.
      wandb_project: Wandb project name.
      wandb_entity: Wandb entity (username or team).
      wandb_group: Wandb group for organizing runs.
      wandb_name: Wandb run name.
      wandb_tags: List of tags for the Wandb run.
      wandb_notes: Notes for the Wandb run.
      wandb_mode: Wandb mode ('online', 'offline', or 'disabled').
      wandb_config: Configuration dictionary to log to Wandb.
      init_wandb: Whether to initialize wandb in this logger. Only set to True
                 for the first logger (typically the learner logger).

    Returns:
      A logger object that responds to logger.write(some_dict).
    """
    del steps_key
    if not print_fn:
        print_fn = logging.info
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]

    if save_data:
        loggers.append(
            csv.CSVLogger(label=label, directory_or_file=save_dir, add_uid=add_uid)
        )

    # Add Wandb logger if requested
    if use_wandb:
        if not WANDB_LOGGER_AVAILABLE:
            print(
                f"Warning: Wandb logging requested but wandb_logger not available for {label}"
            )
        else:
            try:
                wandb_logger = WandbLogger(
                    label=label,
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config,
                    group=wandb_group,
                    name=wandb_name,
                    tags=wandb_tags,
                    notes=wandb_notes,
                    mode=wandb_mode,
                    init_wandb=init_wandb,
                )
                loggers.append(wandb_logger)
                print(f"Wandb logging enabled for {label}")
            except Exception as e:
                print(f"Warning: Failed to initialize Wandb logger for {label}: {e}")

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)
    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, time_delta)

    return logger
