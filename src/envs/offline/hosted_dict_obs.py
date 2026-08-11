"""Hosted-Minari Dict-observation adapter fills (MiniGrid/BabyAI family).

Hosted MiniGrid/BabyAI datasets record the NATIVE symbolic observation —
``Dict(direction, image (T+1, V, V, 3) uint8, mission)`` — which the frozen
loaders in ``minari_loader.py`` cannot ingest (``torch.as_tensor`` dies on the
object-dtype Dict array; see docs/minari_adoption_report.md, fourrooms spike).
These fills adapt that family: the symbolic ``image`` is flattened to a
``(V*V*3,)`` float32 vector per step and everything else mirrors the frozen
fills' transition contract ``{obs, actions, rewards, next_obs, dones}``.

DELIBERATELY SEPARATE from ``fill_replay_buffer_from_minari`` /
``fill_sequence_buffer_from_minari`` (same duplication discipline as the
flat/sequence split): the frozen fills' bytes stay untouched, and this module
never runs for Box-obs datasets — the runner dispatches here only when the
loaded dataset's ``observation_space`` is a ``gymnasium.spaces.Dict``.

Encoding contract: the flattened float32 symbolic vector must match what the
``minigrid_symbolic`` env wrapper (``make_minigrid_symbolic_env`` ->
``ImgObsWrapper`` [+ ``FullyObsWrapper``] -> ``FlattenObservation`` ->
GymnasiumEnv's vector float32 cast) delivers at eval time — raw index values,
NO /255 (meaningless on symbolic object/color/state indices). ``direction`` and
``mission`` are dropped: the egocentric view already encodes orientation, and
the fixed-task datasets carry a constant mission string.

No ``mask_indices`` / ``load_u``: hosted datasets carry no confounder U and the
masked-obs axis is defined on our generated vector envs. The runner raises
before dispatching here if either is requested.
"""

from __future__ import annotations

import numpy as np
import torch

from .minari_loader import load_minari_dataset


def _flat_symbolic_obs(episode, dataset_id: str) -> torch.Tensor:
    """(T+1, V, V, 3) uint8 symbolic ``image`` -> (T+1, V*V*3) float32."""
    obs = episode.observations
    if not isinstance(obs, dict) or "image" not in obs:
        raise ValueError(
            f"Dict-obs adapter fill on dataset '{dataset_id}' expects episode "
            "observations to be a dict with an 'image' key (MiniGrid/BabyAI "
            f"family); got {type(obs).__name__}. Use the standard fill for "
            "Box-obs datasets."
        )
    img = np.asarray(obs["image"])
    if img.dtype == object:
        # Minari 0.5.x HDF5 writer/reader asymmetry: the WRITER JPEG-encodes any
        # uint8 (T, H, W, 1|3) array (data-based heuristic), but the READER only
        # decodes when the SPACE is >=32x32 ("we don't consider images if
        # smaller, e.g. minigrid") — so small symbolic grids written by 0.5.x
        # come back as an object array of JPEG byte-blobs. Decode them the same
        # way minari's own big-image path does. The hosted MiniGrid/BabyAI
        # datasets are UNAFFECTED (written raw by minari 0.4.x); this branch
        # serves locally-written datasets/fixtures, where the JPEG round-trip is
        # LOSSY — the loss happened at write time and cannot be undone here.
        import io

        from PIL import Image

        img = np.stack(
            [np.asarray(Image.open(io.BytesIO(b)), dtype=np.uint8) for b in img]
        )
    return torch.as_tensor(img.reshape(img.shape[0], -1), dtype=torch.float32)


def fill_replay_buffer_from_minari_dict(
    dataset_id: str, buffer, device: torch.device
) -> int:
    """Flat ``ReplayBuffer`` fill for a Dict-obs (MiniGrid/BabyAI) hosted dataset.

    Same sink semantics as ``fill_replay_buffer_from_minari``: one 5-key
    transition per step, ``dones = terminations | truncations``. Returns the
    number of transitions added.
    """
    dataset = load_minari_dataset(dataset_id)
    n_added = 0
    for episode in dataset.iterate_episodes():
        obs = _flat_symbolic_obs(episode, dataset_id)
        actions = torch.as_tensor(episode.actions)
        rewards = torch.as_tensor(episode.rewards, dtype=torch.float32)
        terminations = torch.as_tensor(episode.terminations, dtype=torch.bool)
        truncations = torch.as_tensor(episode.truncations, dtype=torch.bool)
        dones = (terminations | truncations).float()
        for t in range(rewards.shape[0]):
            buffer.add(
                {
                    "obs": obs[t],
                    "actions": actions[t],
                    "rewards": rewards[t],
                    "next_obs": obs[t + 1],
                    "dones": dones[t],
                }
            )
            n_added += 1
    return n_added


def fill_sequence_buffer_from_minari_dict(
    dataset_id: str, seq_buffer, device: torch.device
) -> int:
    """Episode-grouped fill for a Dict-obs hosted dataset (recurrent learners).

    Mirrors ``fill_sequence_buffer_from_minari``'s sink: one buffer episode per
    Minari episode keyed by enumerate index, ``add(ep_idx, transition)`` then
    ``mark_episode_end(ep_idx)``, so ``sample_sequences`` yields same-episode
    windows. Returns the number of transitions added.
    """
    dataset = load_minari_dataset(dataset_id)
    n_added = 0
    for ep_idx, episode in enumerate(dataset.iterate_episodes()):
        obs = _flat_symbolic_obs(episode, dataset_id)
        actions = torch.as_tensor(episode.actions)
        rewards = torch.as_tensor(episode.rewards, dtype=torch.float32)
        terminations = torch.as_tensor(episode.terminations, dtype=torch.bool)
        truncations = torch.as_tensor(episode.truncations, dtype=torch.bool)
        dones = (terminations | truncations).float()
        for t in range(rewards.shape[0]):
            seq_buffer.add(
                ep_idx,
                {
                    "obs": obs[t],
                    "actions": actions[t],
                    "rewards": rewards[t],
                    "next_obs": obs[t + 1],
                    "dones": dones[t],
                },
            )
            n_added += 1
        seq_buffer.mark_episode_end(ep_idx)
    return n_added
