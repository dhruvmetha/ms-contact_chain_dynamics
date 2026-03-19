from omegaconf import OmegaConf


class Logger:
    """Thin wrapper around WandB for episode logging."""

    def __init__(self, cfg):
        self._enabled = cfg.logging.use_wandb
        self._run = None
        if self._enabled:
            import wandb

            self._run = wandb.init(
                project=cfg.logging.project,
                group=cfg.logging.group,
                config={
                    "seed": cfg.seed,
                    "task": OmegaConf.to_container(cfg.task, resolve=True),
                    "runtime": OmegaConf.to_container(cfg.runtime, resolve=True),
                    "run": OmegaConf.to_container(cfg.run, resolve=True),
                },
            )

    def log_episode(self, metrics: dict, step: int):
        if self._enabled:
            import wandb

            wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict):
        if self._enabled:
            import wandb

            for k, v in metrics.items():
                wandb.run.summary[k] = v

    def finish(self):
        if self._enabled and self._run is not None:
            self._run.finish()
