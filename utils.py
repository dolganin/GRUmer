import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def save_rewards(rewards, path):
    df = pd.DataFrame({'reward': rewards})
    df.to_csv(path, index=False)

def save_config(config_dict, path):
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def save_plot(reward_lists, labels, out_path):
    plt.figure(figsize=(10, 5))
    for r, l in zip(reward_lists, labels):
        plt.plot(r, label=l)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
