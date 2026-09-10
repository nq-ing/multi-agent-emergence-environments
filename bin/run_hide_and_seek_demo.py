#!/usr/bin/env python3
"""Run the original Hide-and-Seek quadrant environment and record a video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from mae_envs.envs.hide_and_seek import make_env
from ma_policy.numpy_policy import NumpyHideAndSeekPolicy


def build_env(horizon: int):
    return make_env(
        n_hiders=2,
        n_seekers=2,
        grab_box=True,
        grab_out_of_vision=False,
        grab_selective=False,
        grab_exclusive=False,
        lock_box=True,
        lock_type="all_lock_team_specific",
        lock_out_of_vision=False,
        n_substeps=15,
        horizon=horizon,
        scenario="quadrant",
        prep_fraction=0.4,
        rew_type="joint_zero_sum",
        restrict_rect=[0.1, 0.1, 5.9, 5.9],
        p_door_dropout=0.5,
        quadrant_game_hider_uniform_placement=True,
        n_boxes=2,
        box_only_z_rot=True,
        boxid_obs=False,
        n_ramps=1,
        lock_ramp=False,
        penalize_objects_out=True,
        n_food=0,
        n_lidar_per_agent=30,
        prep_obs=True,
    )


def make_camera():
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [3.0, 3.0, 0.0]
    camera.distance = 8.0
    camera.azimuth = 0.0
    camera.elevation = -70.0
    return camera


def sample_action(env):
    return {key: np.asarray(value) for key, value in env.action_space.sample().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/hide_and_seek_pretrained.gif"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--policy", choices=("pretrained", "random"),
                        default="pretrained")
    parser.add_argument("--weights", type=Path,
                        default=Path("examples/hide_and_seek_quadrant.npz"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = build_env(args.steps)
    env.seed(args.seed)
    observation = env.reset()
    policy = None
    if args.policy == "pretrained":
        policy = NumpyHideAndSeekPolicy(args.weights)
        policy.reset(env.unwrapped.n_agents)

    sim = env.unwrapped.sim
    renderer = mujoco.Renderer(
        sim.model._model, height=args.height, width=args.width)
    camera = make_camera()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_reward = np.zeros(env.unwrapped.n_agents, dtype=np.float64)
    frame_means = []
    if args.output.suffix.lower() == ".gif":
        writer = imageio.get_writer(
            args.output, mode="I", duration=1000 / args.fps, loop=0)
    else:
        writer = imageio.get_writer(
            args.output, fps=args.fps, codec="libx264", quality=8,
            macro_block_size=None)

    try:
        for step in range(args.steps + 1):
            renderer.update_scene(sim.data._data, camera=camera)
            frame = renderer.render().copy()
            frame_means.append(float(frame.mean()))
            writer.append_data(frame)

            if step == args.steps:
                break

            action = policy.act(observation) if policy is not None else sample_action(env)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if (step + 1) % 20 == 0 or done:
                print(
                    f"step={step + 1:03d}/{args.steps} "
                    f"sim_time={sim.data.time:6.2f}s "
                    f"reward={total_reward.tolist()} "
                    f"discard={info.get('discard_episode', False)}",
                    flush=True,
                )
            if done and step + 1 < args.steps:
                raise RuntimeError("Environment ended before the requested horizon")
    finally:
        writer.close()
        renderer.close()

    summary = {
        "environment": "OpenAI multi-agent hide-and-seek quadrant",
        "agents": {"hiders": 2, "seekers": 2},
        "steps": args.steps,
        "policy": args.policy,
        "sim_time_seconds": float(sim.data.time),
        "total_reward": total_reward.tolist(),
        "mean_pixel_value": float(np.mean(frame_means)),
        "non_black_video": bool(np.mean(frame_means) > 1.0),
        "video": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
