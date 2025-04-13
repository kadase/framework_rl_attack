import gymnasium as gym
import cv2
import numpy as np
from gymnasium import spaces
import torch

class OnlineAttackEnv(gym.Env):
    def __init__(self, target_model=None, demo_mode=False, use_target_model=False):
        super().__init__()
        self.target_model = target_model  # Целевая модель (опционально)
        self.demo_mode = demo_mode  # Демо-режим
        self.use_target_model = use_target_model  # Использовать ли целевую модель для расчета награды
        self.max_retries = 3
        self.max_episode_steps = 100
        self.current_step = 0
        self.last_processed_frame = None
        self.state = None  # Инициализируем атрибут state

        # Пространства наблюдений и действий
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.float32  # Формат (height, width, channels)
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3, 84, 84), dtype=np.float32  # Формат (channels, height, width)
        )

        # Инициализация камеры/демо-режима
        if not self.demo_mode:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("Не удалось открыть камеру!")
        else:
            self.demo_frames = [np.random.rand(84, 84, 3) for _ in range(100)]
            self.current_frame_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0

        if self.demo_mode:
            self.current_frame_idx = 0
            observation = self._preprocess(self.demo_frames[self.current_frame_idx])
        else:
            for _ in range(self.max_retries):
                ret, frame = self.cap.read()
                if ret:
                    observation = self._preprocess(frame)
                    break
                else:
                    print("Повторная попытка...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
            else:
                raise ValueError("Не удалось получить кадр с камеры после нескольких попыток")

        # Убедитесь, что наблюдение возвращается в формате (H, W, C)
        self.last_processed_frame = observation.copy()
        self.state = observation  # Инициализируем state текущим кадром
        return observation, {}


    def step(self, action):
        self.current_step += 1

        # Получаем кадр
        if self.demo_mode:
            if self.current_frame_idx >= len(self.demo_frames):
                frame = np.zeros((84, 84, 3))
                terminated = True
            else:
                frame = self.demo_frames[self.current_frame_idx]
                self.current_frame_idx += 1
                terminated = False
        else:
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Не удалось получить кадр с камеры")
            terminated = False

        # Обработка кадра
        processed_frame = self._preprocess(frame)
        if np.isnan(processed_frame).any() or np.isinf(processed_frame).any():
            raise ValueError("Обработанный кадр содержит NaN или бесконечные значения!")

        # Применяем возмущение
        perturbed_frame = self.apply_perturbation(action)
        if np.isnan(perturbed_frame.numpy()).any() or np.isinf(perturbed_frame.numpy()).any():
            raise ValueError("Возмущенный кадр содержит NaN или бесконечные значения!")

        # Расчет награды
        reward = self._calculate_reward(perturbed_frame)


        # Проверка завершения эпизода
        truncated = self.current_step >= self.max_episode_steps
        terminated = terminated or truncated

        # Обновляем last_processed_frame
        self.last_processed_frame = processed_frame.copy()

        # Убедитесь, что наблюдение возвращается в формате (H, W, C)
        return perturbed_frame.numpy(), float(reward), terminated, truncated, {}

    def _preprocess(self, frame):
        # Добавляем размытие, чтобы снизить шум
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Уменьшаем размер изображения до 84x84
        resized = cv2.resize(blurred.astype(np.uint8), (84, 84))

        # Нормализуем изображение в диапазон [0, 1]
        resized = resized / 255.0

        return resized  # (H, W, C)


    def apply_perturbation(self, action):
        if self.state is None:
            raise ValueError("State is not initialized. Call reset() first.")
        
        # Проверка размерности action
        if action.shape != (3, 84, 84):
            raise ValueError(f"Action shape {action.shape} does not match expected shape (3, 84, 84)!")
        
        # Преобразуем self.state и action в torch.Tensor
        state_tensor = torch.tensor(self.state, dtype=torch.float32)  # Формат (H, W, C)
        action_tensor = torch.tensor(action, dtype=torch.float32)  # Формат (C, H, W)
        
        # Применяем возмущение
        perturbed_state = state_tensor + action_tensor.permute(1, 2, 0)  # Приводим action к формату (H, W, C)
        perturbed_state = torch.clamp(perturbed_state, 0, 1)  # Обрезаем значения в диапазоне [0, 1]
        
        # Обновляем состояние среды
        self.state = perturbed_state.numpy()  # Обновляем состояние среды (numpy.ndarray)
        return perturbed_state  # Возвращаем perturbed_state как torch.Tensor

    def _calculate_reward(self, perturbed_frame):
        if self.use_target_model and self.target_model is not None:
            with torch.no_grad():
                input_tensor = torch.tensor(perturbed_frame, dtype=torch.float32).unsqueeze(0)
                if input_tensor.shape[-1] == 3:
                    input_tensor = input_tensor.permute(0, 3, 1, 2)
                input_tensor = torch.clamp(input_tensor, 0.0, 1.0)

                prediction = self.target_model(input_tensor)
                probs = torch.nn.functional.softmax(prediction, dim=1)[0]
                new_class = torch.argmax(probs).item()

            if self.last_processed_frame is not None:
                original_tensor = torch.tensor(self.last_processed_frame, dtype=torch.float32).unsqueeze(0)
                if original_tensor.shape[-1] == 3:
                    original_tensor = original_tensor.permute(0, 3, 1, 2)
                original_tensor = torch.clamp(original_tensor, 0.0, 1.0)
                original_pred = self.target_model(original_tensor)
                original_probs = torch.nn.functional.softmax(original_pred, dim=1)[0]
                old_class = torch.argmax(original_probs).item()

                # Компоненты награды
                class_changed = 1.0 if old_class != new_class else 0.0
                confidence_drop = original_probs[old_class] - probs[old_class]
                l2_penalty = torch.norm(input_tensor - original_tensor)

                # Весовые коэффициенты
                w1, w2, w3 = 1.5, 1.0, 0.3

                reward = (w1 * class_changed) + (w2 * confidence_drop.item()) - (w3 * l2_penalty.item())
            else:
                reward = 0.0
        else:
            # Без модели — используем L2 разницу как суррогат
            if self.last_processed_frame is None:
                reward = 0.0
            else:
                reward = np.mean(np.abs(perturbed_frame.numpy() - self.last_processed_frame)) + 1e-8

        return float(reward)



    def close(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

# Регистрация среды
gym.register(
    id="OnlineAttackEnv-v0",
    entry_point="core.environment:OnlineAttackEnv",
    kwargs={"demo_mode": False}
)