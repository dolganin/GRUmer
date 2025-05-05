from dm_control import suite
import torch
import numpy as np
import imageio.v2 as imageio  # используем imageio для gif

def record_rollout(filepath="rollout.gif"):
    env = suite.load("cheetah", "run")
    ts = env.reset()
    frames = []

    # Dummy policy: просто вперёд
    for _ in range(500):
        frame = env.physics.render(height=240, width=320, camera_id=0)
        frames.append(frame)
        action = np.ones(env.action_spec().shape) * 0.1
        ts = env.step(action)
        if ts.last():
            break

    # Сохраняем в .gif
    imageio.mimsave(filepath, frames, fps=30)
