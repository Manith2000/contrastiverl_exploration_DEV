"""Defines distributed contrastive RL agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.jax import utils
from acme.utils import loggers
from contrastive import builder
from contrastive import config as contrastive_config
from contrastive import distributed_layout
from contrastive import networks
from contrastive import utils as contrastive_utils

from default import make_default_logger, make_wandb_logger

import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec], networks.ContrastiveNetworks]


class DistributedContrastive(distributed_layout.DistributedLayout):
    """Distributed program definition for contrastive RL."""

    def __init__(
        self,
        environment_factory,
        environment_factory_fixed_goals,
        network_factory,
        config,
        seed,
        num_actors,
        max_number_of_steps=None,
        log_to_bigtable=False,
        log_every=10.0,
        evaluator_factories=None,
    ):
        # Check that the environment-specific parts of the config have been set.
        assert config.max_episode_steps > 0
        assert config.obs_dim > 0

        # Prepare wandb config if enabled
        wandb_config_dict = None
        if config.use_wandb:
            wandb_config_dict = {
                "algorithm": config.alg_name,
                "environment": config.env_name,
                "seed": seed,
                "max_steps": config.max_number_of_steps,
                "num_actors": num_actors,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "actor_learning_rate": config.actor_learning_rate,
                "discount": config.discount,
                "tau": config.tau,
                "use_td": config.use_td,
                "use_cpc": config.use_cpc,
                "repr_dim": config.repr_dim,
                "hidden_layer_sizes": config.hidden_layer_sizes,
            }

        logger_fn = functools.partial(
            make_wandb_logger,
            "learner",
            log_to_bigtable,
            time_delta=log_every,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            save_dir=config.log_dir
            + config.alg_name
            + "_"
            + config.env_name
            + "_"
            + str(seed),
            add_uid=config.add_uid,
            steps_key="learner_steps",
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            wandb_group=config.wandb_group,
            wandb_name=config.wandb_name,
            wandb_tags=config.wandb_tags,
            wandb_notes=config.wandb_notes,
            wandb_mode=config.wandb_mode,
            wandb_config=wandb_config_dict,
            init_wandb=True,
        )  # Initialize wandb in learner logger
        contrastive_builder = builder.ContrastiveBuilder(config, logger_fn=logger_fn)
        if evaluator_factories is None:
            eval_policy_factory = lambda n: networks.apply_policy_and_sample(n, True)
            eval_observers = [
                contrastive_utils.SuccessObserver(),
                contrastive_utils.DistanceObserver(
                    obs_dim=config.obs_dim,
                    start_index=config.start_index,
                    end_index=config.end_index,
                ),
            ]
            evaluator_factories = [
                distributed_layout.default_evaluator_factory(
                    environment_factory=environment_factory_fixed_goals,
                    network_factory=network_factory,
                    policy_factory=eval_policy_factory,
                    log_to_bigtable=log_to_bigtable,
                    observers=eval_observers,
                    save_dir=config.log_dir
                    + config.alg_name
                    + "_"
                    + config.env_name
                    + "_"
                    + str(seed),
                    add_uid=config.add_uid,
                    use_wandb=config.use_wandb,
                    wandb_project=config.wandb_project,
                    wandb_entity=config.wandb_entity,
                    wandb_group=config.wandb_group,
                    wandb_mode=config.wandb_mode,
                )
            ]
            if config.local:
                evaluator_factories = []
        actor_observers = [
            contrastive_utils.SuccessObserver(),
            contrastive_utils.DistanceObserver(
                obs_dim=config.obs_dim,
                start_index=config.start_index,
                end_index=config.end_index,
            ),
        ]
        super().__init__(
            seed=seed,
            environment_factory=environment_factory,
            environment_factory_fixed_goals=environment_factory_fixed_goals,
            network_factory=network_factory,
            builder=contrastive_builder,
            policy_network=networks.apply_policy_and_sample,
            evaluator_factories=evaluator_factories,
            num_actors=num_actors,
            max_number_of_steps=max_number_of_steps,
            prefetch_size=config.prefetch_size,
            log_to_bigtable=log_to_bigtable,
            actor_logger_fn=distributed_layout.get_default_logger_fn(
                log_to_bigtable,
                log_every,
                save_dir=config.log_dir
                + config.alg_name
                + "_"
                + config.env_name
                + "_"
                + str(seed),
                add_uid=config.add_uid,
                use_wandb=config.use_wandb,
                wandb_project=config.wandb_project,
                wandb_entity=config.wandb_entity,
                wandb_group=config.wandb_group,
                wandb_mode=config.wandb_mode,
            ),
            observers=actor_observers,
            checkpointing_config=distributed_layout.CheckpointingConfig(
                save_dir=config.log_dir
                + config.alg_name
                + "_"
                + config.env_name
                + "_"
                + str(seed),
                add_uid=config.add_uid,
            ),
            config=config,
        )
