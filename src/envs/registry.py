from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .base import BaseEnv


@dataclass
class EnvWrapperSpec:
    name: str
    builder: Callable[..., BaseEnv]
    match: Optional[Callable[[str], bool]] = None
    requires_entry_point: bool = False


class EnvWrapperRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, EnvWrapperSpec] = {}
        self._order: list[str] = []

    def register(self, spec: EnvWrapperSpec) -> None:
        key = spec.name.lower()
        self._specs[key] = spec
        if key not in self._order:
            self._order.append(key)

    def get(self, name: str) -> EnvWrapperSpec:
        key = name.lower()
        if key not in self._specs:
            raise KeyError(f"Env wrapper '{name}' not registered")
        return self._specs[key]

    def resolve(
        self, env_id: str, wrapper_name: str | None, env_entry_point: str | None
    ) -> EnvWrapperSpec:
        if wrapper_name and wrapper_name.lower() != "auto":
            return self.get(wrapper_name)

        for key in self._order:
            spec = self._specs[key]
            if spec.match and spec.match(env_id):
                return spec

        if "gymnasium" in self._specs:
            return self._specs["gymnasium"]
        if self._order:
            return self._specs[self._order[0]]
        raise KeyError("No env wrappers registered.")


registry = EnvWrapperRegistry()
_DEFAULTS_REGISTERED = False


def register_default_env_wrappers() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    from .wrappers.custom_env import CustomEnv
    from .wrappers.gymnasium_env import GymnasiumEnv

    def build_gymnasium(**kwargs) -> BaseEnv:
        return GymnasiumEnv(
            env_id=kwargs["env_id"],
            n_envs=kwargs["n_envs"],
            device=kwargs["device"],
            seed=kwargs["seed"],
            render=kwargs.get("render", False),
            record_video=kwargs.get("record_video", False),
            video_path=kwargs.get("video_path"),
        )

    def build_custom(**kwargs) -> BaseEnv:
        return CustomEnv(
            env_id=kwargs["env_id"],
            n_envs=kwargs["n_envs"],
            device=kwargs["device"],
            seed=kwargs["seed"],
            render=kwargs.get("render", False),
            record_video=kwargs.get("record_video", False),
            video_path=kwargs.get("video_path"),
            env_entry_point=kwargs.get("env_entry_point"),
            env_kwargs=kwargs.get("env_kwargs"),
        )

    def build_masked(**kwargs) -> BaseEnv:
        # Builds a gymnasium env and drops the configured observation indices.
        # ``mask_indices`` arrives via ``env_kwargs``; the primary CLI path
        # (``--mask-indices``) wraps in the runner so masking composes on the
        # OUTSIDE of any train-env confounding (see MaskedObservationWrapper).
        from .wrappers.masked import MaskedObservationWrapper

        env_kwargs = kwargs.get("env_kwargs") or {}
        indices = env_kwargs.get("mask_indices")
        if not indices:
            raise ValueError(
                "env-wrapper 'masked' requires env_kwargs['mask_indices'] "
                "(a non-empty list of integer observation indices)."
            )
        base = build_gymnasium(**kwargs)
        return MaskedObservationWrapper(base, tuple(int(i) for i in indices))

    registry.register(
        EnvWrapperSpec(
            name="custom",
            builder=build_custom,
            match=lambda env_id: env_id.lower().startswith(("custom:", "user:")),
            requires_entry_point=True,
        )
    )
    registry.register(
        EnvWrapperSpec(
            name="gymnasium",
            builder=build_gymnasium,
            match=None,
            requires_entry_point=False,
        )
    )
    # Opt-in observation masking (Z-hidden axis). ``match=None`` -> never
    # auto-selected; reachable only via an explicit ``--env-wrapper masked``.
    registry.register(
        EnvWrapperSpec(
            name="masked",
            builder=build_masked,
            match=None,
            requires_entry_point=False,
        )
    )

    _DEFAULTS_REGISTERED = True


def resolve_env_wrapper(
    env_id: str, wrapper_name: str | None, env_entry_point: str | None
) -> EnvWrapperSpec:
    return registry.resolve(env_id, wrapper_name, env_entry_point)


def build_env(
    *,
    env_id: str,
    n_envs: int,
    device,
    seed: int,
    render: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    env_wrapper: str | None = None,
    env_entry_point: str | None = None,
    env_kwargs: Optional[dict] = None,
) -> BaseEnv:
    spec = resolve_env_wrapper(env_id, env_wrapper, env_entry_point)
    resolved_entry_point = env_entry_point
    if spec.requires_entry_point and not resolved_entry_point:
        if env_id.lower().startswith(("custom:", "user:")):
            resolved_entry_point = env_id.split(":", 1)[1]
    if spec.requires_entry_point and not resolved_entry_point:
        raise ValueError(
            f"Env wrapper '{spec.name}' requires env_entry_point to be set."
        )
    return spec.builder(
        env_id=env_id,
        n_envs=n_envs,
        device=device,
        seed=seed,
        render=render,
        record_video=record_video,
        video_path=video_path,
        env_entry_point=resolved_entry_point,
        env_kwargs=env_kwargs or {},
    )
