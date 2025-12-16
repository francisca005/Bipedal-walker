import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import BipedalWalker

class InjuredBipedalWalker(BipedalWalker):
    """
    Versão modificada do BipedalWalker-v3 onde a Perna Direita (Junta 2)
    tem uma falha mecânica e perde força.
    """
    def __init__(self, render_mode=None, hardcore=False):
        # Inicializa a classe pai (BipedalWalker original)
        super().__init__(render_mode=render_mode, hardcore=hardcore)
        
        # Fator de dano: 0.3 significa que a perna só tem 30% da força original
        self.damage_factor = 0.3 

    def step(self, action):
        # A ação é um vetor de 4 valores: [Anca1, Joelho1, Anca2, Joelho2]
        # Perna 1 (Esquerda) = índices 0 e 1
        # Perna 2 (Direita)  = índices 2 e 3
        action = np.array(action, copy=True)
        
        # Aplicar o dano: Reduzir a força da perna direita
        action[2] = action[2] * self.damage_factor
        action[3] = action[3] * self.damage_factor
        
        # Passar a ação modificada para o motor de física original
        obs, reward, terminated, truncated, info = super().step(action)
        
        return obs, reward, terminated, truncated, info