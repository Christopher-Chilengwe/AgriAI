# reinforcement_learning/irrigation_optimizer.py
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import save_to_pkl
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from config import RLConfig  # Custom configuration class
from crop_model import CropGrowthSimulator  # Physics-based crop simulator

class AdvancedIrrigationEnv(gym.Env):
    METADATA = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: RLConfig = RLConfig()):
        super().__init__()
        self.config = config
        self.crop_sim = CropGrowthSimulator(config)
        
        # Continuous action space (water amount in mm)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Enhanced observation space
        self.observation_space = spaces.Dict({
            'soil_moisture': spaces.Box(low=0, high=1, shape=(1,)),
            'crop_growth': spaces.Box(low=0, high=1, shape=(1,)),
            'weather': spaces.Box(
                low=np.array([-20, 0, 0]),
                high=np.array([50, 100, 10]),
                dtype=np.float32
            ),
            'temporal': spaces.Box(
                low=np.array([0, 0]),
                high=np.array([365, 24]),
                dtype=np.float32
            )
        })
        
        self._reset_sim()

    def _reset_sim(self):
        """Initialize simulation state"""
        self.state = {
            'soil_moisture': np.array([0.5]),
            'crop_growth': np.array([0.1]),
            'weather': np.array([25.0, 60.0, 2.0]),  # temp, humidity, wind
            'temporal': np.array([0.0, 12.0])  # day_of_year, hour
        }
        self.sim_step = 0
        self.total_reward = 0.0
        self.water_used = 0.0

    def step(self, action: np.ndarray):
        # Convert action to actual water amount
        water_amount = action[0] * self.config.max_water_per_step
        
        # Update simulation
        self.state = self.crop_sim.update_state(
            self.state,
            water_amount,
            self.sim_step
        )
        
        # Calculate reward components
        reward = self._calculate_reward(water_amount)
        
        # Update episode tracking
        self.sim_step += 1
        self.total_reward += reward
        self.water_used += water_amount
        
        # Check termination
        done = self.sim_step >= self.config.max_steps
        
        return self._get_obs(), reward, done, self._get_info()

    def _calculate_reward(self, water_amount: float) -> float:
        """Multi-objective reward function with penalty terms"""
        growth_reward = self.state['crop_growth'][0] * 100
        water_penalty = water_amount * self.config.water_cost
        moisture_penalty = abs(
            self.state['soil_moisture'][0] - self.config.optimal_moisture
        ) * 10
        return growth_reward - water_penalty - moisture_penalty

    def reset(self):
        self._reset_sim()
        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Return normalized observation"""
        return {
            'soil_moisture': self.state['soil_moisture'],
            'crop_growth': self.state['crop_growth'],
            'weather': self.state['weather'] / np.array([50.0, 100.0, 10.0]),
            'temporal': self.state['temporal'] / np.array([365.0, 24.0])
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            'total_reward': self.total_reward,
            'water_used': self.water_used,
            'growth_stage': self.state['crop_growth'][0]
        }

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step: {self.sim_step}")
            print(f"Soil Moisture: {self.state['soil_moisture'][0]:.2f}")
            print(f"Crop Growth: {self.state['crop_growth'][0]:.2f}")
            print(f"Total Water Used: {self.water_used:.2f} mm")
        elif mode == 'rgb_array':
            return self._generate_visualization()

    def _generate_visualization(self) -> np.ndarray:
        """Create matplotlib visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        # Add visualization code
        plt.close()
        return fig2array(fig)

class IrrigationOptimizer:
    def __init__(self, config: RLConfig = RLConfig()):
        self.config = config
        self.env = self._create_environment()
        self.model = self._initialize_model()

    def _create_environment(self):
        """Create vectorized normalized environment"""
        env = make_vec_env(
            lambda: Monitor(AdvancedIrrigationEnv(self.config)),
            n_envs=self.config.n_envs,
            vec_env_cls=DummyVecEnv
        )
        return VecNormalize(env, norm_obs=True, norm_reward=True)

    def _initialize_model(self):
        """Initialize RL model with custom network"""
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]
            )
        )
        
        return SAC(
            "MultiInputPolicy",
            self.env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            device='auto',
            verbose=1
        )

    def train(self):
        """Execute training pipeline"""
        callbacks = [
            EvalCallback(
                self.env,
                best_model_save_path="./best_model",
                log_path="./logs",
                eval_freq=5000,
                deterministic=True,
                render=False
            ),
            CheckpointCallback(
                save_freq=10000,
                save_path="./checkpoints",
                name_prefix="irrigation_model"
            )
        ]
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            log_interval=4,
            tb_log_name="sac_irrigation"
        )
        
        self._save_final_model()

    def _save_final_model(self):
        """Save model and normalization stats"""
        self.model.save("./trained_models/irrigation_ai")
        stats_path = "./trained_models/vecnormalize.pkl"
        save_to_pkl(stats_path, self.env)

    def evaluate(self, num_episodes: int = 10):
        """Evaluate trained policy"""
        self.env.training = False
        self.env.norm_reward = False
        
        results = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.env.step(action)
            results.append(info[0])
        
        df = pd.DataFrame(results)
        print("\nEvaluation Results:")
        print(f"Average Reward: {df['total_reward'].mean():.2f}")
        print(f"Average Water Used: {df['water_used'].mean():.2f} mm")
        print(f"Final Growth Stage: {df['growth_stage'].mean():.2f}")
        
        self._plot_training_progress()

    def _plot_training_progress(self):
        """Generate training performance plots"""
        # Implementation for plotting metrics from tensorboard logs

def fig2array(fig):
    """Convert matplotlib figure to numpy array"""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())

if __name__ == "__main__":
    config = RLConfig(
        total_timesteps=1_000_000,
        max_water_per_step=20.0,
        optimal_moisture=0.6,
        n_envs=4
    )
    
    optimizer = IrrigationOptimizer(config)
    optimizer.train()
    optimizer.evaluate()
