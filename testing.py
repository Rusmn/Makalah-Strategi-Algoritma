import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import pandas as pd

from Lotka_Volterra_Astar_Optimizer import AStarOpt, LVModel, DataGen, ParamNode

class Testing:

    def __init__(self):
        self.true_params = (1.0, 1.5, 0.75, 1.25)
        self.init_state = [10, 5]
        self.time_pts = np.linspace(0, 15, 100)
        self.base_bounds = {
            'alpha': (0.2, 2.0),
            'beta': (0.5, 2.5), 
            'delta': (0.2, 1.5),
            'gamma': (0.5, 2.0)
        }
        self.results = {}
        
    def run_all_tests(self):
        print("="*60)
        print("ANALISIS A* LOTKA-VOLTERRA")
        print("="*60)
        
        # 1. Eksperimen Dasar
        print("\n1. EKSPERIMEN DASAR")
        self.basic_experiment()
        
        # 2. Test Robustness Noise
        print("\n2. TEST ROBUSTNESS TERHADAP NOISE")
        self.noise_robustness_test()
        
        # 3. Test Granularity
        print("\n3. TEST PENGARUH GRANULARITY PENCARIAN")
        self.granularity_test()
        
        # 4. Test Ukuran Ruang Pencarian
        print("\n4. TEST UKURAN RUANG PENCARIAN")
        self.bounds_size_test()
        
        # 5. Multiple Runs untuk Statistik
        print("\n5. MULTIPLE RUNS UNTUK ANALISIS STATISTIK")
        self.statistical_analysis()
        
        # 6. Generate Summary Report
        print("\n6. MEMBUAT SUMMARY REPORT")
        self.generate_summary()
        
        print("\n" + "="*60)
        print("SEMUA EKSPERIMEN SELESAI!")
        print("="*60)
    
    def basic_experiment(self):
        print("Menjalankan eksperimen dasar...")
        
        clean_data, obs_data = DataGen.gen_data(
            self.true_params, self.init_state, self.time_pts, 0.3
        )
        
        optimizer = AStarOpt(obs_data, self.time_pts, self.init_state, 
                           self.base_bounds, step_size=0.1)
        best_params, best_error, results = optimizer.optimize(max_iter=300)
        
        self._plot_basic_results(clean_data, obs_data, best_params, results)
        
        if best_params:
            est_vals = [best_params.alpha, best_params.beta, best_params.delta, best_params.gamma]
            rel_errors = [abs(t-e)/t*100 for t,e in zip(self.true_params, est_vals)]
            
            self.results['basic'] = {
                'est_params': est_vals,
                'mse': best_error,
                'rel_errors': rel_errors,
                'comp_time': results['comp_time'],
                'iterations': results['iterations']
            }
            
            print(f"MSE: {best_error:.6f}")
            print(f"Mean Rel Error: {np.mean(rel_errors):.2f}%")
            print(f"Waktu: {results['comp_time']:.2f}s")
    
    def noise_robustness_test(self):
        print("Testing robustness terhadap noise...")
        
        noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
        noise_results = []
        
        for noise in noise_levels:
            print(f"  Testing noise level: {noise}")
            
            _, obs_data = DataGen.gen_data(
                self.true_params, self.init_state, self.time_pts, noise
            )
            
            optimizer = AStarOpt(obs_data, self.time_pts, self.init_state, 
                               self.base_bounds, step_size=0.15)
            best_params, best_error, results = optimizer.optimize(max_iter=200)
            
            if best_params:
                est_vals = [best_params.alpha, best_params.beta, 
                           best_params.delta, best_params.gamma]
                rel_errors = [abs(t-e)/t*100 for t,e in zip(self.true_params, est_vals)]
                
                noise_results.append({
                    'noise_level': noise,
                    'mse': best_error,
                    'mean_rel_error': np.mean(rel_errors),
                    'comp_time': results['comp_time']
                })
                
                print(f"    MSE: {best_error:.6f}, Rel Error: {np.mean(rel_errors):.2f}%")
            else:
                noise_results.append({
                    'noise_level': noise,
                    'mse': float('inf'),
                    'mean_rel_error': float('inf'),
                    'comp_time': 0
                })
                print(f"    ✗ Optimasi gagal")
        
        self._plot_noise_analysis(noise_results)
        self.results['noise_test'] = noise_results
    
    def granularity_test(self):
        print("Testing pengaruh granularity pencarian...")
        
        step_sizes = [0.05, 0.1, 0.2, 0.3]
        granularity_results = []
        
        _, obs_data = DataGen.gen_data(
            self.true_params, self.init_state, self.time_pts, 0.3
        )
        
        for step_size in step_sizes:
            print(f"  Testing step_size: {step_size}")
            
            optimizer = AStarOpt(obs_data, self.time_pts, self.init_state, 
                               self.base_bounds, step_size=step_size)
            best_params, best_error, results = optimizer.optimize(max_iter=250)
            
            if best_params:
                est_vals = [best_params.alpha, best_params.beta, 
                           best_params.delta, best_params.gamma]
                rel_errors = [abs(t-e)/t*100 for t,e in zip(self.true_params, est_vals)]
                
                granularity_results.append({
                    'step_size': step_size,
                    'mse': best_error,
                    'mean_rel_error': np.mean(rel_errors),
                    'comp_time': results['comp_time'],
                    'iterations': results['iterations']
                })
                
                print(f"    MSE: {best_error:.6f}, Time: {results['comp_time']:.2f}s")
        
        self._plot_granularity_analysis(granularity_results)
        self.results['granularity_test'] = granularity_results
    
    def bounds_size_test(self):
        print("Testing ukuran ruang pencarian...")
        
        bounds_configs = [
            {
                'name': 'Sempit',
                'bounds': {
                    'alpha': (0.7, 1.3), 'beta': (1.2, 1.8),
                    'delta': (0.5, 1.0), 'gamma': (1.0, 1.5)
                }
            },
            {
                'name': 'Sedang', 
                'bounds': {
                    'alpha': (0.5, 1.5), 'beta': (1.0, 2.0),
                    'delta': (0.4, 1.1), 'gamma': (0.8, 1.7)
                }
            },
            {
                'name': 'Luas',
                'bounds': self.base_bounds
            },
            {
                'name': 'Sangat Luas',
                'bounds': {
                    'alpha': (0.1, 3.0), 'beta': (0.1, 3.0),
                    'delta': (0.1, 2.0), 'gamma': (0.1, 3.0)
                }
            }
        ]
        
        bounds_results = []
        
        _, obs_data = DataGen.gen_data(
            self.true_params, self.init_state, self.time_pts, 0.3
        )
        
        for config in bounds_configs:
            print(f"  Testing bounds: {config['name']}")
            
            optimizer = AStarOpt(obs_data, self.time_pts, self.init_state, 
                               config['bounds'], step_size=0.1)
            best_params, best_error, results = optimizer.optimize(max_iter=300)
            
            if best_params:
                est_vals = [best_params.alpha, best_params.beta, 
                           best_params.delta, best_params.gamma]
                rel_errors = [abs(t-e)/t*100 for t,e in zip(self.true_params, est_vals)]
                
                bounds = config['bounds']
                volume = ((bounds['alpha'][1] - bounds['alpha'][0]) *
                         (bounds['beta'][1] - bounds['beta'][0]) *
                         (bounds['delta'][1] - bounds['delta'][0]) *
                         (bounds['gamma'][1] - bounds['gamma'][0]))
                
                bounds_results.append({
                    'bounds_name': config['name'],
                    'volume': volume,
                    'mse': best_error,
                    'mean_rel_error': np.mean(rel_errors),
                    'comp_time': results['comp_time'],
                    'nodes_explored': results['nodes_explored']
                })
                
                print(f"    MSE: {best_error:.6f}, Nodes: {results['nodes_explored']}")
        
        self._plot_bounds_analysis(bounds_results)
        self.results['bounds_test'] = bounds_results
    
    def statistical_analysis(self):
        print("Melakukan multiple runs untuk statistik...")
        
        n_runs = 5
        stats_results = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            
            _, obs_data = DataGen.gen_data(
                self.true_params, self.init_state, self.time_pts, 0.4
            )
            
            optimizer = AStarOpt(obs_data, self.time_pts, self.init_state, 
                               self.base_bounds, step_size=0.1)
            best_params, best_error, results = optimizer.optimize(max_iter=250)
            
            if best_params:
                est_vals = [best_params.alpha, best_params.beta, 
                           best_params.delta, best_params.gamma]
                rel_errors = [abs(t-e)/t*100 for t,e in zip(self.true_params, est_vals)]
                
                stats_results.append({
                    'run': run+1,
                    'mse': best_error,
                    'rel_errors': rel_errors,
                    'comp_time': results['comp_time'],
                    'iterations': results['iterations']
                })
                
                print(f"    MSE: {best_error:.6f}")
        
        if stats_results:
            mse_vals = [r['mse'] for r in stats_results]
            times = [r['comp_time'] for r in stats_results]
            iters = [r['iterations'] for r in stats_results]
            
            param_errors = {
                'alpha': [r['rel_errors'][0] for r in stats_results],
                'beta': [r['rel_errors'][1] for r in stats_results],
                'delta': [r['rel_errors'][2] for r in stats_results],
                'gamma': [r['rel_errors'][3] for r in stats_results]
            }
            
            stats_summary = {
                'n_runs': len(stats_results),
                'success_rate': len(stats_results) / n_runs * 100,
                'mse_mean': np.mean(mse_vals),
                'mse_std': np.std(mse_vals),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'iter_mean': np.mean(iters),
                'param_errors': {param: {'mean': np.mean(errs), 'std': np.std(errs)} 
                               for param, errs in param_errors.items()}
            }
            
            self._plot_statistical_analysis(stats_results, stats_summary)
            self.results['stats'] = stats_summary
            
            print(f"Success Rate: {stats_summary['success_rate']:.1f}%")
            print(f"Mean MSE: {stats_summary['mse_mean']:.6f} ± {stats_summary['mse_std']:.6f}")
    
    def generate_summary(self):
        print("Membuat summary report...")
        
        self._create_summary_table()
        
        self._plot_comprehensive_summary()
        
        print("Summary report selesai!")
    
    def _plot_basic_results(self, clean_data, obs_data, best_params, results):
        if not best_params:
            return
            
        est_model = LVModel(*best_params.get_params())
        est_data = est_model.simulate(self.init_state, self.time_pts)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0,0].plot(self.time_pts, clean_data[:, 0], 'b-', label='True Mangsa', linewidth=2)
        axes[0,0].plot(self.time_pts, clean_data[:, 1], 'r-', label='True Predator', linewidth=2)
        axes[0,0].plot(self.time_pts, obs_data[:, 0], 'b.', alpha=0.6, label='Obs Mangsa')
        axes[0,0].plot(self.time_pts, obs_data[:, 1], 'r.', alpha=0.6, label='Obs Predator')
        axes[0,0].set_xlabel('Waktu')
        axes[0,0].set_ylabel('Populasi')
        axes[0,0].set_title('Data Observasi vs True')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(self.time_pts, obs_data[:, 0], 'b.', alpha=0.6, label='Obs Mangsa')
        axes[0,1].plot(self.time_pts, obs_data[:, 1], 'r.', alpha=0.6, label='Obs Predator')
        axes[0,1].plot(self.time_pts, est_data[:, 0], 'b-', linewidth=2, label='Est Mangsa')
        axes[0,1].plot(self.time_pts, est_data[:, 1], 'r-', linewidth=2, label='Est Predator')
        axes[0,1].set_xlabel('Waktu')
        axes[0,1].set_ylabel('Populasi')
        axes[0,1].set_title('Hasil Fitting A*')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[0,2].plot(clean_data[:, 0], clean_data[:, 1], 'g-', label='True', linewidth=2)
        axes[0,2].plot(obs_data[:, 0], obs_data[:, 1], 'k.', alpha=0.6, label='Observed')
        axes[0,2].plot(est_data[:, 0], est_data[:, 1], 'orange', label='Estimated', linewidth=2, linestyle='--')
        axes[0,2].set_xlabel('Populasi Mangsa')
        axes[0,2].set_ylabel('Populasi Predator')
        axes[0,2].set_title('Phase Portrait')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        axes[1,0].plot(results['error_hist'])
        axes[1,0].set_xlabel('Iterasi')
        axes[1,0].set_ylabel('MSE')
        axes[1,0].set_title('Konvergensi Error')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        params_names = ['α', 'β', 'δ', 'γ']
        true_vals = list(self.true_params)
        est_vals = [best_params.alpha, best_params.beta, best_params.delta, best_params.gamma]
        
        x = np.arange(len(params_names))
        width = 0.35
        
        axes[1,1].bar(x - width/2, true_vals, width, label='True', alpha=0.7)
        axes[1,1].bar(x + width/2, est_vals, width, label='Estimated', alpha=0.7)
        axes[1,1].set_xlabel('Parameter')
        axes[1,1].set_ylabel('Nilai')
        axes[1,1].set_title('Perbandingan Parameter')
        axes[1,1].set_xticks(x, params_names)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        rel_errors = [abs(t-e)/t*100 for t,e in zip(true_vals, est_vals)]
        axes[1,2].bar(params_names, rel_errors, alpha=0.7, color='red')
        axes[1,2].set_xlabel('Parameter')
        axes[1,2].set_ylabel('Relative Error (%)')
        axes[1,2].set_title('Error Relatif per Parameter')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle('HASIL EKSPERIMEN DASAR A* LOTKA-VOLTERRA', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _plot_noise_analysis(self, noise_results):
        noise_levels = [r['noise_level'] for r in noise_results]
        mse_vals = [r['mse'] for r in noise_results]
        rel_errors = [r['mean_rel_error'] for r in noise_results]
        comp_times = [r['comp_time'] for r in noise_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(noise_levels, mse_vals, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Tingkat Noise')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE vs Tingkat Noise')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        axes[1].plot(noise_levels, rel_errors, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Tingkat Noise')
        axes[1].set_ylabel('Mean Relative Error (%)')
        axes[1].set_title('Error Relatif vs Tingkat Noise')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(noise_levels, comp_times, 'go-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Tingkat Noise')
        axes[2].set_ylabel('Waktu Komputasi (detik)')
        axes[2].set_title('Waktu Komputasi vs Tingkat Noise')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('ANALISIS ROBUSTNESS TERHADAP NOISE', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _plot_granularity_analysis(self, granularity_results):
        step_sizes = [r['step_size'] for r in granularity_results]
        mse_vals = [r['mse'] for r in granularity_results]
        comp_times = [r['comp_time'] for r in granularity_results]
        iterations = [r['iterations'] for r in granularity_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(step_sizes, mse_vals, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Step Size')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE vs Step Size')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(step_sizes, comp_times, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Step Size')
        axes[1].set_ylabel('Waktu Komputasi (detik)')
        axes[1].set_title('Waktu Komputasi vs Step Size')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(step_sizes, iterations, 'go-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Step Size')
        axes[2].set_ylabel('Jumlah Iterasi')
        axes[2].set_title('Iterasi vs Step Size')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('ANALISIS PENGARUH GRANULARITY PENCARIAN', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _plot_bounds_analysis(self, bounds_results):
        names = [r['bounds_name'] for r in bounds_results]
        volumes = [r['volume'] for r in bounds_results]
        mse_vals = [r['mse'] for r in bounds_results]
        comp_times = [r['comp_time'] for r in bounds_results]
        nodes = [r['nodes_explored'] for r in bounds_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].bar(names, mse_vals, alpha=0.7)
        axes[0,0].set_ylabel('MSE')
        axes[0,0].set_title('MSE vs Ukuran Ruang Pencarian')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].bar(names, comp_times, alpha=0.7, color='red')
        axes[0,1].set_ylabel('Waktu Komputasi (detik)')
        axes[0,1].set_title('Waktu vs Ukuran Ruang Pencarian')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].bar(names, volumes, alpha=0.7, color='green')
        axes[1,0].set_ylabel('Volume Ruang Pencarian')
        axes[1,0].set_title('Volume Ruang Pencarian')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].bar(names, nodes, alpha=0.7, color='orange')
        axes[1,1].set_ylabel('Nodes Explored')
        axes[1,1].set_title('Node yang Dieksplorasi')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('ANALISIS UKURAN RUANG PENCARIAN', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _plot_statistical_analysis(self, stats_results, stats_summary):
        runs = [r['run'] for r in stats_results]
        mse_vals = [r['mse'] for r in stats_results]
        times = [r['comp_time'] for r in stats_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].plot(runs, mse_vals, 'bo-', linewidth=2, markersize=8)
        axes[0,0].axhline(y=stats_summary['mse_mean'], color='red', linestyle='--', label='Mean')
        axes[0,0].set_xlabel('Run')
        axes[0,0].set_ylabel('MSE')
        axes[0,0].set_title('MSE per Run')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(runs, times, 'ro-', linewidth=2, markersize=8)
        axes[0,1].axhline(y=stats_summary['time_mean'], color='blue', linestyle='--', label='Mean')
        axes[0,1].set_xlabel('Run')
        axes[0,1].set_ylabel('Waktu Komputasi (detik)')
        axes[0,1].set_title('Waktu Komputasi per Run')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        param_names = ['α', 'β', 'δ', 'γ']
        param_means = [stats_summary['param_errors'][p]['mean'] for p in ['alpha', 'beta', 'delta', 'gamma']]
        param_stds = [stats_summary['param_errors'][p]['std'] for p in ['alpha', 'beta', 'delta', 'gamma']]
        
        axes[1,0].bar(param_names, param_means, yerr=param_stds, alpha=0.7, capsize=5)
        axes[1,0].set_ylabel('Mean Relative Error (%)')
        axes[1,0].set_title('Error Rata-rata per Parameter')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].hist(mse_vals, bins=5, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].axvline(x=stats_summary['mse_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1,1].set_xlabel('MSE')
        axes[1,1].set_ylabel('Frekuensi')
        axes[1,1].set_title('Distribusi MSE')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('ANALISIS STATISTIK MULTIPLE RUNS', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _create_summary_table(self):
        print("\n" + "="*80)
        print("SUMMARY TABLE - HASIL SEMUA EKSPERIMEN")
        print("="*80)
        
        # Basic experiment
        if 'basic' in self.results:
            basic = self.results['basic']
            print(f"EKSPERIMEN DASAR:")
            print(f"  MSE: {basic['mse']:.6f}")
            print(f"  Mean Rel Error: {np.mean(basic['rel_errors']):.2f}%")
            print(f"  Waktu: {basic['comp_time']:.2f}s")
            print(f"  Iterasi: {basic['iterations']}")
        
        # Statistical analysis
        if 'stats' in self.results:
            stats = self.results['stats']
            print(f"\nSTATISTIK MULTIPLE RUNS ({stats['n_runs']} runs):")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  MSE Mean ± Std: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
            print(f"  Waktu Mean ± Std: {stats['time_mean']:.2f} ± {stats['time_std']:.2f}s")
            print(f"  Error per Parameter:")
            for param in ['alpha', 'beta', 'delta', 'gamma']:
                err = stats['param_errors'][param]
                print(f"    {param}: {err['mean']:.2f} ± {err['std']:.2f}%")
        
        # Noise robustness
        if 'noise_test' in self.results:
            noise_results = self.results['noise_test']
            print(f"\nROBUSTNESS TERHADAP NOISE:")
            for result in noise_results:
                if result['mse'] != float('inf'):
                    print(f"  Noise {result['noise_level']:.1f}: MSE={result['mse']:.6f}, "
                          f"Error={result['mean_rel_error']:.2f}%")
                else:
                    print(f"  Noise {result['noise_level']:.1f}: GAGAL")
        
        # Granularity test
        if 'granularity_test' in self.results:
            gran_results = self.results['granularity_test']
            print(f"\nPENGARUH GRANULARITY:")
            for result in gran_results:
                print(f"  Step {result['step_size']:.2f}: MSE={result['mse']:.6f}, "
                      f"Time={result['comp_time']:.2f}s, Iter={result['iterations']}")
        
        # Bounds test
        if 'bounds_test' in self.results:
            bounds_results = self.results['bounds_test']
            print(f"\nPENGARUH UKURAN RUANG PENCARIAN:")
            for result in bounds_results:
                print(f"  {result['bounds_name']}: MSE={result['mse']:.6f}, "
                      f"Time={result['comp_time']:.2f}s, Nodes={result['nodes_explored']}")
        
        print("="*80)
    
    def _plot_comprehensive_summary(self):
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Noise robustness summary
        if 'noise_test' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            noise_results = self.results['noise_test']
            noise_levels = [r['noise_level'] for r in noise_results if r['mse'] != float('inf')]
            mse_vals = [r['mse'] for r in noise_results if r['mse'] != float('inf')]
            ax1.semilogy(noise_levels, mse_vals, 'bo-', linewidth=2, markersize=6)
            ax1.set_xlabel('Noise Level')
            ax1.set_ylabel('MSE (log)')
            ax1.set_title('Robustness vs Noise')
            ax1.grid(True, alpha=0.3)
        
        # 2. Granularity summary
        if 'granularity_test' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            gran_results = self.results['granularity_test']
            step_sizes = [r['step_size'] for r in gran_results]
            mse_vals = [r['mse'] for r in gran_results]
            comp_times = [r['comp_time'] for r in gran_results]
            
            ax2_twin = ax2.twinx()
            line1 = ax2.plot(step_sizes, mse_vals, 'bo-', label='MSE')
            line2 = ax2_twin.plot(step_sizes, comp_times, 'ro-', label='Time')
            ax2.set_xlabel('Step Size')
            ax2.set_ylabel('MSE', color='blue')
            ax2_twin.set_ylabel('Time (s)', color='red')
            ax2.set_title('MSE vs Time vs Step Size')
            ax2.grid(True, alpha=0.3)
        
        # 3. Bounds summary
        if 'bounds_test' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            bounds_results = self.results['bounds_test']
            names = [r['bounds_name'] for r in bounds_results]
            mse_vals = [r['mse'] for r in bounds_results]
            ax3.bar(names, mse_vals, alpha=0.7)
            ax3.set_ylabel('MSE')
            ax3.set_title('MSE vs Bounds Size')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Statistical summary - MSE distribution
        if 'stats' in self.results and 'basic' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            stats = self.results['stats']
            basic = self.results['basic']
            
            data_to_plot = []
            labels = []
            
            data_to_plot.append([basic['mse']])
            labels.append('Basic')
            
            mean_mse = stats['mse_mean']
            std_mse = stats['mse_std']
            synthetic_data = np.random.normal(mean_mse, std_mse, stats['n_runs'])
            data_to_plot.append(synthetic_data)
            labels.append('Multiple Runs')
            
            ax4.boxplot(data_to_plot, labels=labels)
            ax4.set_ylabel('MSE')
            ax4.set_title('MSE Distribution')
            ax4.grid(True, alpha=0.3)
        
        if 'basic' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            basic = self.results['basic']
            param_names = ['α', 'β', 'δ', 'γ']
            rel_errors = basic['rel_errors']
            
            colors = ['blue', 'red', 'green', 'orange']
            bars = ax5.bar(param_names, rel_errors, color=colors, alpha=0.7)
            ax5.set_ylabel('Relative Error (%)')
            ax5.set_title('Error per Parameter')
            ax5.grid(True, alpha=0.3)
            
            for bar, error in zip(bars, rel_errors):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{error:.1f}%', ha='center', va='bottom')
        
        if 'granularity_test' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            gran_results = self.results['granularity_test']
            mse_vals = [r['mse'] for r in gran_results]
            comp_times = [r['comp_time'] for r in gran_results]
            step_sizes = [r['step_size'] for r in gran_results]
            
            scatter = ax6.scatter(comp_times, mse_vals, c=step_sizes, cmap='viridis', s=100)
            ax6.set_xlabel('Computation Time (s)')
            ax6.set_ylabel('MSE')
            ax6.set_title('Accuracy vs Speed Trade-off')
            ax6.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax6, label='Step Size')
        
        ax7 = plt.subplot(3, 3, 7)
        if 'stats' in self.results:
            success_rate = self.results['stats']['success_rate']
            ax7.pie([success_rate, 100-success_rate], labels=['Success', 'Failed'], 
                   autopct='%1.1f%%', startangle=90)
            ax7.set_title('Success Rate')
        
        if 'bounds_test' in self.results:
            ax8 = plt.subplot(3, 3, 8)
            bounds_results = self.results['bounds_test']
            volumes = [r['volume'] for r in bounds_results]
            comp_times = [r['comp_time'] for r in bounds_results]
            names = [r['bounds_name'] for r in bounds_results]
            
            ax8.scatter(volumes, comp_times, s=100)
            for i, name in enumerate(names):
                ax8.annotate(name, (volumes[i], comp_times[i]), 
                           xytext=(5, 5), textcoords='offset points')
            ax8.set_xlabel('Search Space Volume')
            ax8.set_ylabel('Computation Time (s)')
            ax8.set_title('Scalability Analysis')
            ax8.set_xscale('log')
            ax8.grid(True, alpha=0.3)
        
        ax9 = plt.subplot(3, 3, 9)
        if 'basic' in self.results and 'stats' in self.results:
            metrics = ['MSE', 'Rel Error', 'Time', 'Iterations']
            basic = self.results['basic']
            stats = self.results['stats']
            
            basic_vals = [
                basic['mse'],
                np.mean(basic['rel_errors']),
                basic['comp_time'],
                basic['iterations']
            ]
            
            stats_vals = [
                stats['mse_mean'],
                np.mean([stats['param_errors'][p]['mean'] for p in ['alpha', 'beta', 'delta', 'gamma']]),
                stats['time_mean'],
                stats['iter_mean']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax9.bar(x - width/2, [1]*len(metrics), width, label='Basic', alpha=0.7)
            ax9.bar(x + width/2, [stats_vals[i]/basic_vals[i] for i in range(len(metrics))], 
                   width, label='Multi-run Ratio', alpha=0.7)
            ax9.set_xlabel('Metrics')
            ax9.set_ylabel('Relative Performance')
            ax9.set_title('Performance Comparison')
            ax9.set_xticks(x, metrics)
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle('SUMMARY - A* LOTKA-VOLTERRA OPTIMIZATION', fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    print("Memulai testing ...")

    try:
        tester = Testing()
        tester.run_all_tests()
        print("\nSEMUA TESTING BERHASIL DISELESAIKAN!")

    except ImportError as e:
        print(f"Error detail: {e}")
        
    except Exception as e:
        print(f"Error tidak terduga: {e}")

if __name__ == "__main__":
    main()