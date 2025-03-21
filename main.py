import os
import sys
import pandas as pd

sys.path.append(os.path.relpath("First-Come-First-Serve-scheduling"))
sys.path.append(os.path.relpath("Priority-Scheduling"))
sys.path.append(os.path.relpath("Round-Robin-scheduling"))
sys.path.append(os.path.relpath("Shortest-Job-First-scheduling"))

from FCFS import simulate_fcfs_algorithm
from SJF_np import simulate_sjf_np_algorithm
from SJF_p import simulate_sjf_p_algorithm
from RR import simulate_rr_algorithm
from priority_p import simulate_priority_p_algorithm
from priority_np import simulate_priority_np_algorithm

# Import PPO Model
from stable_baselines3 import PPO
from RL_Scheduler import CPUSchedulingEnv

def load_processes_from_csv(file_path):
    data = pd.read_csv(file_path)
    processes = []
    
    for _, row in data.iterrows():
        processes.append({
            "arrival": row["arrival_time"],  
            "burst": row["burst_time"],      
            "priority": row["priority"]      
        })
    
    return processes

def calculate_metrics(process_list):
        total_wait = 0
        total_turnaround = 0
        total_response = 0
        total_time = 0
        n = len(process_list)

        for p in process_list:
            turnaround = p["finish_time"] - p["arrival"]
            wait = turnaround - p["burst"]
            response = p["start_time"] - p["arrival"]

            total_wait += wait
            total_turnaround += turnaround
            total_response += response
            total_time += p["burst"]

        print(f"\nPPO Scheduling Results:")
        print(f"Throughput = {n / total_time:.4f}")
        print(f"Average waiting time = {total_wait / n:.4f}")
        print(f"Average turn around time = {total_turnaround / n:.4f}")
        print(f"Average response time = {total_response / n:.4f}")

if __name__ == "__main__":
    data = pd.read_csv("db/data_set.csv")

    # Run traditional scheduling algorithms
    print("\nRunning Traditional Scheduling Algorithms:")
    simulate_fcfs_algorithm(data)
    simulate_sjf_np_algorithm(data)
    simulate_sjf_p_algorithm(data)
    quantum = 1
    simulate_rr_algorithm(data, quantum)
    simulate_priority_np_algorithm(data)
    simulate_priority_p_algorithm(data)

    # Load processes for RL scheduling
    processes = load_processes_from_csv("db/data_set.csv")

    # Initialize RL environment
    env = CPUSchedulingEnv(processes)

    # Load trained PPO model or train a new one
    # Check if model exists before loading
    if os.path.exists("ppo_scheduler.zip"):
        print("\nLoading trained PPO scheduler...")
        model = PPO.load("ppo_scheduler")
    else:
        print("\nTraining PPO scheduler...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save("ppo_scheduler")
        print("\nPPO scheduler trained and saved.")

    # Run RL-based scheduling
    print("\nRunning RL-Based Scheduling...")
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    calculate_metrics(env.finished)

    print("\nPPO Scheduling Completed.")
