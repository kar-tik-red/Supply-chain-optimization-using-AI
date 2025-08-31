import numpy as np
import pandas as pd
import math
import random
import gym
from gym import spaces

from stable_baselines3 import PPO



CSV_FILE = "/Users/sharingan/Desktop/Skills/AI_project/augmented_data.csv"  

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find {CSV_FILE}. Place the file in the same folder or update the path."
    )

# Basic checks to ensure columns exist; adapt as needed:
required_cols = ["xCoord", "yCoord", "scheduled_delivery_hours"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV must contain a column named '{col}'")

NUM_CUSTOMERS = len(df)
print(f"Loaded dataset with {NUM_CUSTOMERS} orders/customers.")


customer_coords = df[["xCoord", "yCoord"]].to_numpy()




warehouse_coords = {
    "A": (10, 10),
    "B": (90, 10),
    "C": (50, 90)
}

WAREHOUSES = list(warehouse_coords.keys())  # ["A", "B", "C"]

def distance_from_warehouse(warehouse_label, cust_idx):
    """
    Euclidean distance from the specified warehouse to the cust_idx in the dataset.
    Adjust if using lat/long or a different distance formula.
    """
    wx, wy = warehouse_coords[warehouse_label]
    cx, cy = customer_coords[cust_idx]
    return math.sqrt((wx - cx)**2 + (wy - cy)**2)



POP_SIZE = 30
MAX_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

def create_individual():
    
    return [random.choice(WAREHOUSES) for _ in range(NUM_CUSTOMERS)]

def evaluate_individual(ind):
    """
    Evaluate cost of an individual's plan:
    - Base cost: sum of distances * cost_factor
    - Lateness penalty if delivery_time > deadline
    This is a toy version. In your scenario, incorporate capacity constraints or shipping modes if needed.
    """
    total_cost = 0.0
    late_deliveries = 0
    
    cost_factor = 2.0  
    speed = 30.0       

    for cust_idx, wh_label in enumerate(ind):
        dist = distance_from_warehouse(wh_label, cust_idx)
        cost = dist * cost_factor
        delivery_time = dist / speed
        
        scheduled_delivery_hours = df.loc[cust_idx, "scheduled_delivery_hours"]
        if delivery_time > scheduled_delivery_hours:
            late_deliveries += 1
        
        total_cost += cost
    
    penalty_for_late = 100.0
    total_cost += late_deliveries * penalty_for_late
    return total_cost

def mutate(ind):
    """Randomly alter some warehouse assignments."""
    for i in range(NUM_CUSTOMERS):
        if random.random() < MUTATION_RATE:
            ind[i] = random.choice(WAREHOUSES)
    return ind

def crossover(parent1, parent2):
    """One-point crossover."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]
    point = random.randint(1, NUM_CUSTOMERS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def select(population, fitnesses):
    """Tournament selection: pick the better of two random individuals."""
    idx1, idx2 = random.sample(range(len(population)), 2)
    if fitnesses[idx1] < fitnesses[idx2]:
        return population[idx1]
    else:
        return population[idx2]

def run_ga():
    """
    Main GA routine. Returns the best plan + cost after MAX_GENERATIONS.
    """
    population = [create_individual() for _ in range(POP_SIZE)]
    
    for gen in range(MAX_GENERATIONS):
        fitnesses = [evaluate_individual(ind) for ind in population]
        best_idx = np.argmin(fitnesses)
        best_fit = fitnesses[best_idx]
        print(f"[GA] Gen {gen} | Best Cost: {best_fit:.2f}")
        
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            new_pop.append(c2)
        
        population = new_pop[:POP_SIZE]
    
    
    fitnesses = [evaluate_individual(ind) for ind in population]
    best_idx = np.argmin(fitnesses)
    return population[best_idx], fitnesses[best_idx]



class SupplyChainEnv(gym.Env):
    
    
    def __init__(self, df, best_plan):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.best_plan = best_plan
        self.num_customers = len(df)
        
        # Discrete actions: 3 options
        self.action_space = spaces.Discrete(3)
        
        # Observation space: (dist, time_left, traffic_factor, cost_so_far)
        self.observation_space = spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32)
        
        # Cost/time constants
        self.base_cost_factor = 2.0
        self.base_speed = 30.0
        
        self.reset()

    def reset(self):
        self.order_indices = np.random.permutation(self.num_customers)
        self.current_idx = 0
        self.cost_so_far = 0.0
        self.late_deliveries = 0
        return self._get_obs()

    def _get_obs(self):
        if self.current_idx >= self.num_customers:
            return np.array([0, 0, 0, self.cost_so_far], dtype=np.float32)
        
        cust_id = self.order_indices[self.current_idx]
        wh_label = self.best_plan[cust_id]
        dist = distance_from_warehouse(wh_label, cust_id)
        
        time_left = self.df.loc[cust_id, "scheduled_delivery_hours"]
        traffic_factor = np.random.uniform(1.0, 2.0)
        
        return np.array([dist, time_left, traffic_factor, self.cost_so_far], dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        
        if self.current_idx >= self.num_customers:
            done = True
            return self._get_obs(), reward, done, {}
        
        obs = self._get_obs()
        dist, time_left, traffic_factor, c_so_far = obs
        
        cost = dist * self.base_cost_factor * traffic_factor
        delivery_time = dist / self.base_speed
        
        if action == 1:
            # Expedite: +50% cost, -30% time
            cost *= 1.5
            delivery_time *= 0.7
        elif action == 2:
            # Switch to nearest warehouse
            new_wh = self._find_nearest_warehouse(self.order_indices[self.current_idx])
            new_dist = distance_from_warehouse(new_wh, self.order_indices[self.current_idx])
            cost = new_dist * self.base_cost_factor * traffic_factor
            delivery_time = new_dist / self.base_speed
        
        if delivery_time > time_left:
            reward -= 100.0
            self.late_deliveries += 1
        
        reward -= cost
        self.cost_so_far += cost
        
        self.current_idx += 1
        done = (self.current_idx >= self.num_customers)
        
        return self._get_obs(), reward, done, {}

    def _find_nearest_warehouse(self, cust_idx):
        best_label = None
        best_dist = float('inf')
        cx, cy = customer_coords[cust_idx]
        for wh_label, (wx, wy) in warehouse_coords.items():
            d = math.sqrt((wx - cx)**2 + (wy - cy)**2)
            if d < best_dist:
                best_dist = d
                best_label = wh_label
        return best_label


def simulate_ga_only(best_plan):
    """
    Evaluate cost & lateness with random traffic if we just follow best_plan
    with no RL adaptation.
    """
    total_cost = 0.0
    total_late = 0
    cost_factor = 2.0
    speed = 30.0
    
    order_indices = np.random.permutation(NUM_CUSTOMERS)
    for cust_idx in order_indices:
        wh_label = best_plan[cust_idx]
        dist = distance_from_warehouse(wh_label, cust_idx)
        traffic_factor = np.random.uniform(1.0, 2.0)
        
        cost = dist * cost_factor * traffic_factor
        delivery_time = dist / speed
        scheduled_delivery_hours = df.loc[cust_idx, "scheduled_delivery_hours"]
        
        if delivery_time > scheduled_delivery_hours:
            total_late += 1
        
        total_cost += cost
    
    return total_cost, total_late

def main():
    
    best_plan, best_plan_cost = run_ga()
    print(f"\n[MAIN] GA final best cost (no random disruptions): {best_plan_cost:.2f}")
    
    # Evaluate GA in random traffic scenario (GA-only)
    ga_only_cost, ga_only_late = simulate_ga_only(best_plan)
    print(f"[MAIN] GA-Only (random traffic) => Cost: {ga_only_cost:.2f}, Late: {ga_only_late}")

    
    env = SupplyChainEnv(df, best_plan)
    
    # (A) Quick test with a random policy
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # random action
        action = env.action_space.sample()
        obs, reward, done, _info = env.step(action)
        total_reward += reward
    
    ga_rl_cost = env.cost_so_far
    ga_rl_late = env.late_deliveries
    print(f"[MAIN] GA+RL (random policy) => Cost: {ga_rl_cost:.2f}, Late: {ga_rl_late}, Reward: {total_reward:.2f}")

    # (B) Actual RL training with stable-baselines3 (uncomment if installed)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    
    # Evaluate trained policy
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        total_reward += reward
    
    ga_rl_cost = env.cost_so_far
    ga_rl_late = env.late_deliveries
    print(f"[MAIN] GA+RL (PPO) => Cost: {ga_rl_cost:.2f}, Late: {ga_rl_late}, Reward: {total_reward:.2f}")
    

    # ---------------------------
    # 3) Final Comparison
    # ---------------------------
    print("\n[RESULTS]")
    print(f"GA-Only: Cost={ga_only_cost:.2f}, Late={ga_only_late}")
    print(f"GA+RL:   Cost={ga_rl_cost:.2f}, Late={ga_rl_late}")
    
if __name__ == "__main__":
    main()