import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import enode

# ニューラルネットワークの定義
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ハイパーパラメータ
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
target_update = 10
memory_size = 10000
input_dim = 1 + max(len(enode.nodes[node][0]) for node in enode.nodes) + len(enode.times)  # 位置1、最大の行動可能場所数、時間帯
output_dim = max(len(enode.nodes[node][0]) for node in enode.nodes)  # 行動可能場所の数

# 経験再生メモリ
memory = deque(maxlen=memory_size)

# ネットワークの初期化
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# ε-greedyポリシーによる行動選択
def select_action(state, epsilon):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 状態をテンソルに変換し、バッチ次元を追加
    if random.random() < epsilon:
        return random.randint(0, output_dim - 1)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).argmax().item()

# 経験の保存
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# ミニバッチによる学習
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    
    batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32)
    batch_action = torch.tensor(batch_action, dtype=torch.int64)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
    batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32)
    batch_done = torch.tensor(batch_done, dtype=torch.float32)
    
    current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(batch_next_state).max(1)[0]
    expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))
    
    loss = criterion(current_q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 環境の設定
def initialize_state(node_id, time_id):
    state = np.zeros(input_dim)
    state[0] = node_id
    adjacent_nodes, _, times = enode.nodes[node_id]
    for i, adj_node in enumerate(adjacent_nodes):
        state[i + 1] = adj_node
    state[-len(times):] = [1 if t == time_id else 0 for t in range(len(times))]  # ワンホットベクトルとして時間帯を設定
    return state

# 報酬の計算（混雑度を考慮しない）
def calculate_reward(current_node, next_node):
    return 1 if next_node == goal_node else -1  # 目的地に到達した場合の報酬、通常移動のペナルティ

# 最短ルートの取得
def get_optimal_route(start_node, goal_node, time_id):
    route = [start_node]
    current_node = start_node
    while current_node != goal_node:
        state = initialize_state(current_node, time_id)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        next_node = enode.nodes[current_node][0][action]
        route.append(next_node)
        current_node = next_node
        if len(route) > 100:  # ループを防止
            print("ループの可能性があります。ルートを見直してください。")
            break
    return route

# 学習ループ
num_episodes = 1000
start_node = 0  # スタートノード
goal_node = 11  # ゴールノード

episode_rewards = []
episode_steps = []
final_q_values = {}

for episode in range(num_episodes):
    time_id = 0  # 朝の時間帯を選択
    state = initialize_state(start_node, time_id)
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:  # 100回移動したら失敗
        action = select_action(state, epsilon)
        next_node = int(state[action + 1])
        next_state = initialize_state(next_node, time_id)
        reward = calculate_reward(int(state[0]), next_node)
        total_reward += reward
        done = next_node == goal_node
        store_experience(state, action, reward, next_state, done)
        state = next_state
        optimize_model()
        steps += 1
    
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if episode % 10 == 0:
        print(f'Episode {episode}: Steps = {steps}, Total Reward = {total_reward}')

    # 各エピソードの最後に各ノードのQ値を取得
    for node_id in enode.nodes:
        state = initialize_state(node_id, time_id)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor).numpy()
        final_q_values[node_id] = q_values

# 学習結果の表示
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_steps)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

plt.subplot(1, 2, 2)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')

plt.tight_layout()
plt.savefig('nomal_results.png')

# 各ノードの最終Q値の表示
for node_id, q_values in final_q_values.items():
    print(f'Node {node_id}: Q-values = {q_values}')
    best_action = np.argmax(q_values)
    next_node = enode.nodes[node_id][0][best_action]
    print(f' -> Best next node: {next_node}')

# 学習後の最短ルートを取得して表示
optimal_route = get_optimal_route(start_node, goal_node, 0)  # 朝の時間帯を選択
print(f'Optimal route from node {start_node} to node {goal_node}: {optimal_route}')

