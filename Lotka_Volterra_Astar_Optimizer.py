import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

@dataclass
class ParamNode:
    alpha: float
    beta: float  
    delta: float
    gamma: float
    g_cost: float = 0.0
    h_cost: float = 0.0
    f_cost: float = 0.0
    parent: Optional['ParamNode'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def get_params(self) -> Tuple[float, float, float, float]:
        return (self.alpha, self.beta, self.delta, self.gamma)

class LVModel:
    
    def __init__(self, alpha: float, beta: float, delta: float, gamma: float):
        self.alpha = alpha  # laju pertumbuhan mangsa
        self.beta = beta    # tingkat predasi
        self.delta = delta  # efisiensi konversi
        self.gamma = gamma  # laju kematian predator
    
    def equations(self, state: List[float], t: float) -> List[float]:
        x, y = state
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return [dxdt, dydt]
    
    def simulate(self, init_state: List[float], time_pts: np.ndarray) -> np.ndarray:
        solution = odeint(self.equations, init_state, time_pts)
        return solution

class DataGen:
    
    @staticmethod
    def gen_data(true_params: Tuple[float, float, float, float], 
                 init_state: List[float], 
                 time_pts: np.ndarray, 
                 noise_lvl: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        alpha, beta, delta, gamma = true_params
        model = LVModel(alpha, beta, delta, gamma)
        clean_data = model.simulate(init_state, time_pts)
        
        noise = np.random.normal(0, noise_lvl, clean_data.shape)
        noisy_data = clean_data + noise
        noisy_data = np.maximum(noisy_data, 0.1)  
        
        return clean_data, noisy_data

class AStarOpt:
    
    def __init__(self, obs_data: np.ndarray, time_pts: np.ndarray, 
                 init_state: List[float], param_bounds: dict, 
                 step_size: float = 0.1):
        self.obs_data = obs_data
        self.time_pts = time_pts
        self.init_state = init_state
        self.param_bounds = param_bounds
        self.step_size = step_size
        self.best_error = float('inf')
        self.best_params = None
        
    def calc_mse(self, params: Tuple[float, float, float, float]) -> float:
        try:
            alpha, beta, delta, gamma = params
            model = LVModel(alpha, beta, delta, gamma)
            predicted = model.simulate(self.init_state, self.time_pts)
            
            mse = np.mean((self.obs_data - predicted) ** 2)
            return mse
        except:
            return 1e6  
    
    def heuristic_func(self, node: ParamNode) -> float:
        data_var = np.var(self.obs_data)
        min_error = np.sqrt(data_var) * 0.1
        
        ref_alpha = (self.param_bounds['alpha'][0] + self.param_bounds['alpha'][1]) / 2
        ref_beta = (self.param_bounds['beta'][0] + self.param_bounds['beta'][1]) / 2
        ref_delta = (self.param_bounds['delta'][0] + self.param_bounds['delta'][1]) / 2
        ref_gamma = (self.param_bounds['gamma'][0] + self.param_bounds['gamma'][1]) / 2
        
        dist_penalty = (abs(node.alpha - ref_alpha) + abs(node.beta - ref_beta) + 
                       abs(node.delta - ref_delta) + abs(node.gamma - ref_gamma)) * 0.01
        
        return min_error + dist_penalty
    
    def get_neighbors(self, node: ParamNode) -> List[ParamNode]:
        neighbors = []
        steps = [-self.step_size, self.step_size]
        
        for a_step in steps:
            for b_step in steps:
                for d_step in steps:
                    for g_step in steps:
                        new_alpha = node.alpha + a_step
                        new_beta = node.beta + b_step
                        new_delta = node.delta + d_step
                        new_gamma = node.gamma + g_step
                        
                        if (self.param_bounds['alpha'][0] <= new_alpha <= self.param_bounds['alpha'][1] and
                            self.param_bounds['beta'][0] <= new_beta <= self.param_bounds['beta'][1] and
                            self.param_bounds['delta'][0] <= new_delta <= self.param_bounds['delta'][1] and
                            self.param_bounds['gamma'][0] <= new_gamma <= self.param_bounds['gamma'][1]):
                            
                            neighbor = ParamNode(new_alpha, new_beta, new_delta, new_gamma)
                            neighbor.parent = node
                            neighbors.append(neighbor)
        
        return neighbors
    
    def optimize(self, max_iter: int = 500, tolerance: float = 1e-6) -> Tuple[ParamNode, float, dict]:
        start_alpha = (self.param_bounds['alpha'][0] + self.param_bounds['alpha'][1]) / 2
        start_beta = (self.param_bounds['beta'][0] + self.param_bounds['beta'][1]) / 2
        start_delta = (self.param_bounds['delta'][0] + self.param_bounds['delta'][1]) / 2
        start_gamma = (self.param_bounds['gamma'][0] + self.param_bounds['gamma'][1]) / 2
        
        start_node = ParamNode(start_alpha, start_beta, start_delta, start_gamma)
        start_node.h_cost = self.heuristic_func(start_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        open_list = [start_node]
        closed_set = set()
        iterations = 0
        error_history = []
        start_time = time.time()
        
        print("Memulai optimasi A* untuk estimasi parameter...")
        
        while open_list and iterations < max_iter:
            current = heapq.heappop(open_list)
            current_key = (round(current.alpha, 3), round(current.beta, 3), 
                         round(current.delta, 3), round(current.gamma, 3))
            
            if current_key in closed_set:
                continue
                
            closed_set.add(current_key)
            
            current_error = self.calc_mse(current.get_params())
            error_history.append(current_error)
            
            if current_error < self.best_error:
                self.best_error = current_error
                self.best_params = current
                print(f"Iter {iterations}: MSE = {current_error:.6f}")
            
            if current_error < tolerance:
                print(f"Konvergensi pada iterasi {iterations}")
                break
            
            neighbors = self.get_neighbors(current)
            
            for neighbor in neighbors:
                neighbor_key = (round(neighbor.alpha, 3), round(neighbor.beta, 3), 
                              round(neighbor.delta, 3), round(neighbor.gamma, 3))
                
                if neighbor_key in closed_set:
                    continue
                
                neighbor.g_cost = current.g_cost + self.step_size
                neighbor.h_cost = self.heuristic_func(neighbor)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                
                heapq.heappush(open_list, neighbor)
            
            iterations += 1
            
            if iterations % 50 == 0:
                print(f"Iter {iterations}, Best MSE: {self.best_error:.6f}")
        
        end_time = time.time()
        
        results = {
            'iterations': iterations,
            'comp_time': end_time - start_time,
            'error_hist': error_history,
            'final_error': self.best_error,
            'nodes_explored': len(closed_set)
        }
        
        return self.best_params, self.best_error, results
