import numpy as np
import gymnasium as gym
import random
import time
import matplotlib.pyplot as plt

# Configurar el entorno FrozenLake con slippery=True
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

alpha = 0.5      # Tasa de aprendizaje más alta
gamma = 0.9      # Factor de descuento ligeramente menor
epsilon = 1.0    # Explorar al inicio
epsilon_decay = 0.997  # Reducir exploración más lento
epsilon_min = 0.1      # Mantener exploración mínima decente
episodes = 10000  # Aumentar el número de episodios

# Inicializar la Q-table con ceros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Lista para almacenar las recompensas de cada episodio
rewards = []

# Entrenamiento del agente
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Política epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(q_table[state])  # Explotación

        # Tomar acción y recibir recompensa
        next_state, reward, done, _, _ = env.step(action)

        # Actualización de la Q-table con la ecuación de Q-learning
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward

    # Guardar la recompensa obtenida en este episodio
    rewards.append(total_reward)

    # Reducir epsilon para favorecer explotación sobre exploración con el tiempo
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Mostrar progreso
    if (episode + 1) % 500 == 0:
        print(f"Episodio {episode + 1}/{episodes}, Epsilon: {epsilon:.4f}")

print("Entrenamiento completado.\n")

# Graficar la evolución de recompensas
plt.plot(rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.title('Recompensa obtenida por episodio')
plt.show()

# Evaluación del agente entrenado
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")  # Cargar entorno con visualización
state, _ = env.reset()
done = False

print("Ejecutando agente entrenado...")

while not done:
    action = np.argmax(q_table[state])
    state, _, done, _, _ = env.step(action)
    time.sleep(0.5)  # Pausa para ver el movimiento

env.close()
