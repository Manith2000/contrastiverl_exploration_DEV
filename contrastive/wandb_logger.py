"""Weights & Biases logger for Acme."""

import time
from typing import Any, Mapping, Optional

from acme.utils.loggers import base

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class WandbLogger(base.Logger):
    """Logs to Weights & Biases.

    This logger wraps wandb.log() and is compatible with Acme's logger interface.
    """

    def __init__(
        self,
        label: str,
        project: str = "contrastive-rl",
        entity: Optional[str] = None,
        config: Optional[dict] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        init_wandb: bool = True,
    ):
        """Initialize the Wandb logger.

        Args:
            label: Label for the logger (e.g., 'learner', 'actor', 'evaluator').
            project: Wandb project name.
            entity: Wandb entity (username or team name).
            config: Configuration dictionary to log to wandb.
            group: Group name for organizing runs.
            name: Run name (if None, wandb will generate one).
            tags: List of tags for the run.
            notes: Notes for the run.
            mode: Wandb mode ('online', 'offline', or 'disabled').
            init_wandb: Whether to initialize wandb in this logger. Set to False
                       if wandb is already initialized elsewhere.
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self._label = label
        self._time = time.time()

        # Initialize wandb run if requested
        if init_wandb:
            # Check if wandb is already initialized
            if wandb.run is None:
                self._run = wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    group=group,
                    name=name,
                    tags=tags,
                    notes=notes,
                    mode=mode,
                    reinit=False,  # Don't allow multiple inits
                )
            else:
                self._run = wandb.run
        else:
            self._run = wandb.run

        if self._run is None:
            raise RuntimeError(
                "Wandb run not initialized. Either set init_wandb=True or "
                "initialize wandb before creating the logger."
            )

    def write(self, data: Mapping[str, Any]) -> None:
        """Writes data to wandb.

        Args:
            data: A dictionary of key-value pairs to log.
        """
        if self._run is None:
            return

        # Prefix all keys with the label to avoid conflicts between
        # different loggers (learner, actor, evaluator)
        prefixed_data = {f"{self._label}/{key}": value for key, value in data.items()}

        # Extract step information if available
        step = data.get("steps", None)
        if step is None:
            step = data.get("learner_steps", None)
        if step is None:
            step = data.get("actor_steps", None)

        # Log to wandb
        if step is not None:
            wandb.log(prefixed_data, step=int(step))
        else:
            wandb.log(prefixed_data)

    def close(self) -> None:
        """Closes the logger.

        Note: This does NOT call wandb.finish() because multiple loggers
        may be writing to the same wandb run. Call wandb.finish() explicitly
        when all logging is complete.
        """
        pass


def finish_wandb():
    """Finish the wandb run. Call this at the end of training."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
