import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0, log_file="training_log.json"):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_file = log_file

    def _on_step(self) -> bool:
        if hasattr(self.model, "ep_info_buffer"):
            for episode_info in self.model.ep_info_buffer:
                reward = episode_info.get("r")
                length = episode_info.get("l")

                if reward is not None and length is not None:
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)

        # Показываем метрику каждые 10_000 шагов
        if self.num_timesteps % 10_000 == 0 and len(self.episode_rewards) >= 10:
            last_rewards = self.episode_rewards[-10:]
            last_lengths = self.episode_lengths[-10:]

            print(f"\n[{self.num_timesteps} steps]")
            print(f"  Avg Reward (last 10): {np.mean(last_rewards):.2f}")
            print(f"  Max Reward (last 10): {np.max(last_rewards):.2f}")
            print(f"  Min Reward (last 10): {np.min(last_rewards):.2f}")
            print(f"  Avg Episode Length : {np.mean(last_lengths):.2f}")

            self.save_logs()

        return True

    def save_logs(self):
        if not self.episode_rewards:
            return

        logs = {
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths]
        }

        try:
            with open(self.log_file, "w") as f:
                json.dump(logs, f, indent=2)
            print(f"Logs saved to {self.log_file}")
        except Exception as e:
            print(f"Failed to save logs: {e}")

    def _on_training_end(self) -> None:
        print("Training ended. Saving final logs...")
        self.save_logs()
