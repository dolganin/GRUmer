import yaml
import os
from train_plan2explore import train_plan2explore
from utils import save_rewards, save_plot
from multiprocessing import Process

def run_experiment(name, cfg, path):
    print(f"Запускаем {name}...")
    rewards = train_plan2explore(cfg)
    save_rewards(rewards, path)
    print(f"{name} завершён.")
    return rewards

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    cfg_gru = cfg.copy()
    cfg_gru["use_gru"] = True

    cfg_mlp = cfg.copy()
    cfg_mlp["use_gru"] = False

    p1 = Process(target=run_experiment, args=("Plan2Explore-GRU", cfg_gru, "results/rewards_gru.csv"))
    p2 = Process(target=run_experiment, args=("Plan2Explore-MLP", cfg_mlp, "results/rewards_mlp.csv"))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Отрисовка графика, если оба файла существуют
    try:
        import pandas as pd
        rewards_gru = pd.read_csv("results/rewards_gru.csv")["reward"].tolist()
        rewards_mlp = pd.read_csv("results/rewards_mlp.csv")["reward"].tolist()

        save_plot(
            [rewards_gru, rewards_mlp],
            ["Plan2Explore-GRU", "Plan2Explore-MLP"],
            "results/comparison.png",
            colors=["#1f77b4", "#ff7f0e"]
        )
        print("График сохранён в results/comparison.png")
    except Exception as e:
        print("Ошибка при построении графика:", e)
