# import numpy as np
# import gym
# from gym import spaces
# from stable_baselines3 import PPO

# class CPUSchedulingEnv(gym.Env):
#     def __init__(self, processes):
#         super(CPUSchedulingEnv, self).__init__()
#         self.processes = processes
#         self.num_processes = len(processes)
        
#         # Define state space: arrival time, burst time, priority
#         self.observation_space = spaces.Box(low=-1, high=100, shape=(3000,), dtype=np.float32)
        
#         # Define action space: which process to schedule next
#         self.action_space = spaces.Discrete(self.num_processes)
        
#     def generate_processes(self):
#         """
#         Generate a list of processes for scheduling.
#         Each process has an arrival time, burst time, and priority.
#         """
#         # Sample process data (replace with real dataset later)
#         process_list = [
#             {"arrival": 0, "burst": 5, "priority": 1},
#             {"arrival": 2, "burst": 3, "priority": 2},
#             {"arrival": 4, "burst": 2, "priority": 3},
#             {"arrival": 6, "burst": 4, "priority": 1}
#         ]

#         return process_list

#     def reset(self):
#         self.queue = self.generate_processes()
#         self.action_space = spaces.Discrete(len(self.queue)) 
#         return self._get_obs()

#     def step(self, action):
#         if action >= len(self.queue):  
#             action = np.random.randint(0, len(self.queue))  # Pick a valid random process instead

#         process = self.queue.pop(action)

#         # Compute reward (example: lower burst time = better reward)
#         reward = -process["burst"]  # PPO will learn to minimize burst time

#         done = len(self.queue) == 0  # If all processes are scheduled, terminate

#         return self._get_obs(), reward, done, {}


#     def _get_obs(self):
#         obs = []
#         for p in self.queue:
#             obs.extend([p["arrival"], p["burst"], p["priority"]])

#         # Ensure the observation is always the correct size
#         max_size = self.observation_space.shape[0]  # Expected size (e.g., 3000)
        
#         if len(obs) < max_size:
#             obs += [-1] * (max_size - len(obs))  # Pad with -1 for missing values
#         elif len(obs) > max_size:
#             obs = obs[:max_size]  # Truncate extra values

#         return np.array(obs, dtype=np.float32)


import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

class CPUSchedulingEnv(gym.Env):
    def __init__(self, processes):
        super().__init__()
        self.original_processes = processes
        self.num_processes = len(processes)
        self.observation_space = spaces.Box(low=-1, high=100, shape=(self.num_processes * 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_processes)

    def reset(self):
        self.queue = sorted(self.original_processes.copy(), key=lambda p: p["arrival"])
        self.finished = []
        self.current_time = 0
        self.action_space = spaces.Discrete(len(self.queue))
        return self._get_obs()

    def step(self, action):
        # Get available processes (arrived at current_time)
        available = [p for p in self.queue if p["arrival"] <= self.current_time]

        if not available:
            # No job is ready, advance time to next job
            self.current_time = min(p["arrival"] for p in self.queue)
            available = [p for p in self.queue if p["arrival"] <= self.current_time]

        
        if action >= len(available):
            action = np.random.randint(0, len(available))

        selected_process = available[action]

        # Remove selected process from main queue
        self.queue.remove(selected_process)

        # Schedule
        selected_process["start_time"] = self.current_time
        self.current_time += selected_process["burst"]
        selected_process["finish_time"] = self.current_time

        self.finished.append(selected_process)

        
        # turnaround = selected_process["finish_time"] - selected_process["arrival"]
        # # Reward = negative of total time in system (to minimize delay)
        # reward = -turnaround / 100.0 # normalize the reward to scale down 
        # # reward = - (0.5 * turnaround + 0.3 * wait + 0.2 * response) #multi-objective reward

        wait = selected_process["start_time"] - selected_process["arrival"]
        response = wait  # same in non-preemptive scheduling
        turnaround = selected_process["finish_time"] - selected_process["arrival"]
        # Better reward: penalize wait and response time more directly
        reward = - (0.6 * wait + 0.3 * turnaround + 0.1 * response) / 100.0

        #print(f"Reward: {reward}, Wait: {wait}, Turnaround: {turnaround}")

        done = len(self.queue) == 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = []
        for p in self.queue:
            # Normalize time units
            obs.extend([
                p["arrival"] / 100.0,  
                p["burst"] / 10.0,
                p["priority"] / 1000.0
            ])

        max_size = self.observation_space.shape[0]
        if len(obs) < max_size:
            obs += [-1] * (max_size - len(obs))
        else:
            obs = obs[:max_size]

        return np.array(obs, dtype=np.float32)

