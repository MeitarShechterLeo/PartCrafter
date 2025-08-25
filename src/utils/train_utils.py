from src.utils.typing_utils import *

import os
from omegaconf import OmegaConf

from torch import optim
from torch.optim import lr_scheduler
from diffusers.training_utils import *
from diffusers.optimization import get_scheduler

# https://github.com/huggingface/diffusers/pull/9812: fix `self.use_ema_warmup`
class MyEMAModel(EMAModel):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = True,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        self.shadow_params = {n: p.clone().detach() for n, p in named_parameters}

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> "MyEMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.named_parameters(), model_cls=model_cls, model_config=model.config, foreach=foreach)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.named_parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            # cur_decay_value = (1 + step) / (10 + step)
            cur_decay_value = self.decay

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]):
        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        if self.foreach:
            params_grad = {name: param for name, param in named_parameters if param.requires_grad}
            s_params_grad = {name: self.shadow_params[name] for name, param in named_parameters if param.requires_grad}

            if len(params_grad) < len(named_parameters):
                torch._foreach_copy_(
                    [self.shadow_params[name] for name, param in named_parameters if name not in params_grad],
                    [param for name, param in named_parameters if name not in params_grad],
                    non_blocking=True,
                )

            torch._foreach_sub_(
                list(s_params_grad.values()), torch._foreach_sub(list(s_params_grad.values()), list(params_grad.values())), alpha=one_minus_decay
            )

        else:
            for name, param in named_parameters:
                if name not in self.shadow_params:
                    print(f"Warning: {name} not found in EMA. Skipping.")
                    continue
                
                if param.requires_grad:
                    self.shadow_params[name].sub_(one_minus_decay * (self.shadow_params[name] - param))
                else:
                    self.shadow_params[name].copy_(param)

    def copy_to(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.foreach:
            torch._foreach_copy_(
                [param.data for name, param in named_parameters],
                [self.shadow_params[name].to(param.device).data for name, param in named_parameters],
            )
        else:
            for name, param in named_parameters:
                param.data.copy_(self.shadow_params[name].to(param.device).data)

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """

        self.shadow_params = {n: p.pin_memory() for n, p in self.shadow_params.items()}

    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = {
            n: p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            else p.to(device=device, non_blocking=non_blocking)
            for n, p in self.shadow_params.items()
        }

    def store(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = {name: param.detach().cpu().clone() for name, param in named_parameters}

    def restore(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")

        assert len(self.temp_stored_params) == len(named_parameters), "Number of parameters to restore does not match the number of stored parameters"
        
        if self.foreach:
            torch._foreach_copy_(
                [param.data for _, param in named_parameters], [self.temp_stored_params[name].data for name, _ in named_parameters]
            )
        else:
            for name, param in named_parameters:
                param.data.copy_(self.temp_stored_params[name].data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            if not isinstance(shadow_params, dict):
                raise ValueError("shadow_params must be a dict")
            if not all(isinstance(p, torch.Tensor) for p in shadow_params.values()):
                raise ValueError("shadow_params must all be Tensors")

            for name, param in shadow_params.items():
                if name in self.shadow_params:
                    self.shadow_params[name].data.copy_(param.data)
                else:
                    print(f"Warning: {name} found in state_dict but not in the EMA params. Skipping.")

            
def get_configs(yaml_path: str, cli_configs: List[str]=[], **kwargs) -> DictConfig:
    yaml_configs = OmegaConf.load(yaml_path)
    cli_configs = OmegaConf.from_cli(cli_configs)

    configs = OmegaConf.merge(yaml_configs, cli_configs, kwargs)
    OmegaConf.resolve(configs)  # resolve ${...} placeholders
    return configs

def get_optimizer(name: str, params: Parameter, **kwargs) -> Optimizer:
    if name == "adamw":
        return optim.AdamW(params=params, **kwargs)
    else:
        raise NotImplementedError(f"Not implemented optimizer: {name}")

def get_lr_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    if name == "one_cycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs["max_lr"],
            total_steps=kwargs["total_steps"],
            pct_start=kwargs["pct_start"],
        )
    elif name == "cosine_warmup":
        return get_scheduler(
            "cosine", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant_warmup":
        return get_scheduler(
            "constant_with_warmup", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant":
        return lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda _: 1)
    elif name == "linear_decay":
        return lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: max(0., 1. - epoch / kwargs["total_epochs"]),
        )
    else:
        raise NotImplementedError(f"Not implemented lr scheduler: {name}")

def save_experiment_params(
    args: Namespace, 
    configs: DictConfig, 
    save_dir: str
) -> Dict[str, Any]:
    params = OmegaConf.merge(configs, {"args": {k: str(v) for k, v in vars(args).items()}})
    OmegaConf.save(params, os.path.join(save_dir, "params.yaml"))
    return dict(params)


def save_model_architecture(model: Module, save_dir: str) -> None:
    num_buffers = sum(b.numel() for b in model.buffers())
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f"Number of buffers: {num_buffers}\n" +\
        f"Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n" +\
        f"Model architecture:\n{model}"

    with open(os.path.join(save_dir, "model.txt"), "w") as f:
        f.write(message)