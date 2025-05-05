import numpy as np
import torch

def compute_intrinsic_reward(world_models, obs, act):
    preds = [wm(obs, act)[0].detach().numpy() for wm in world_models]
    std = np.std(preds, axis=0)
    return np.mean(std, axis=1)
