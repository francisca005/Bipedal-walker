import gymnasium as gym
import my_envs # Importar o ambiente da perna partida
from stable_baselines3 import PPO, SAC, TD3
import argparse
import os

def train(algo, env_id, timesteps, seed, run_name):
    # Configurar pastas
    log_path = f"logs/{run_name}"
    save_path = f"models/{run_name}/{algo}_{seed}"
    os.makedirs(f"models/{run_name}", exist_ok=True)
    
    print(f"--> A treinar {algo} no ambiente {env_id} (Seed {seed})")
    
    # Criar ambiente
    env = gym.make(env_id)
    
    # Selecionar Classe
    if algo == "PPO": model_cls = PPO
    elif algo == "SAC": model_cls = SAC
    elif algo == "TD3": model_cls = TD3
    else: raise ValueError("Algoritmo desconhecido")
    
    # Criar e Treinar
    model = model_cls("MlpPolicy", env, verbose=1, tensorboard_log="logs/", seed=seed)
    model.learn(total_timesteps=timesteps, tb_log_name=f"{run_name}_{algo}")
    
    # Guardar
    model.save(save_path)
    print(f"Modelo guardado em {save_path}")
    env.close()

if __name__ == "__main__":
    # Permite correr no terminal: python scripts/train_grid.py --algo PPO --env InjuredBipedalWalker-v0
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="PPO, SAC, TD3")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3", help="ID do ambiente")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Timesteps totais")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--name", type=str, default="experiment", help="Nome da experiência")
    
    args = parser.parse_args()
    
    train(args.algo, args.env, args.steps, args.seed, args.name)