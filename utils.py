import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_rewards(rewards, path):
    df = pd.DataFrame({'reward': rewards})
    df.to_csv(path, index=False)

def save_plot(reward_lists, labels, out_path, colors=None):
    means = [np.mean(r) for r in reward_lists]
    stds = [np.std(r) for r in reward_lists]
    maxes = [np.max(r) for r in reward_lists]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    if colors is None:
        colors = ['gray'] * len(labels)
    ax.bar(x - width/2, means, width, yerr=stds, capsize=5, label='Mean', color=colors)
    ax.bar(x + width/2, maxes, width, label='Max', color=colors, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
