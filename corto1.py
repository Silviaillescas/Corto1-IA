import numpy as np
import gymnasium as gym
import random
import time
import matplotlib.pyplot as plt
import os

# Configuración del entorno FrozenLake con mapa aleatorio
env = gym.make("FrozenLake-v1", desc=None, is_slippery=True, render_mode=None)

# Hiperparámetros
alpha = 0.5        # Tasa de aprendizaje
gamma = 0.9        # Factor de descuento
epsilon = 1.0      # Probabilidad inicial de exploración
epsilon_decay = 0.997  # Tasa de reducción de epsilon
epsilon_min = 0.1  # Mínimo valor de epsilon
episodes = 10000   # Número de episodios de entrenamiento
q_table_file = "q_table.npy"  # Archivo para guardar la Q-table

# Inicializar Q-table (cargar si existe)
if os.path.exists(q_table_file):
    q_table = np.load(q_table_file)
    print("Q-table cargada correctamente.")
else:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Listas para almacenar métricas
rewards = []
average_rewards = []

# Entrenamiento del agente
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Estrategia epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(q_table[state])  # Explotación

        # Ejecutar acción y obtener recompensa
        next_state, reward, done, _, _ = env.step(action)

        # Actualizar la Q-table con la ecuación de Q-learning
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward

    # Guardar recompensa obtenida en este episodio
    rewards.append(total_reward)

    # Reducir epsilon gradualmente
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Calcular promedio de recompensas cada 500 episodios
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(rewards[-500:])
        average_rewards.append(avg_reward)
        print(f"Episodio {episode + 1}/{episodes}, Epsilon: {epsilon:.4f}, Promedio de recompensa: {avg_reward:.3f}")

# Guardar la Q-table entrenada
np.save(q_table_file, q_table)
print("Entrenamiento completado y Q-table guardada.\n")

# Graficar la evolución de recompensas
plt.plot(rewards, alpha=0.5, label="Recompensas por episodio")
plt.plot(range(500, episodes+1, 500), average_rewards, marker="o", linestyle="--", color="r", label="Promedio cada 500 episodios")
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.title('Evolución del aprendizaje del agente')
plt.legend()
plt.show()

# Evaluación del agente entrenado
env = gym.make("FrozenLake-v1", desc=None, is_slippery=True, render_mode="human")
state, _ = env.reset()
done = False

print("Ejecutando agente entrenado...")

while not done:
    action = np.argmax(q_table[state])  # Tomar la mejor acción aprendida
    state, _, done, _, _ = env.step(action)
    time.sleep(0.5)  # Pausa para visualización

env.close()
