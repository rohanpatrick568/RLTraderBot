"""
Hyperparameter Search Script for PPO Trading Bot Model Size Optimization

This script performs a hyperparameter search to determine the optimal model architecture
for the PPO trading bot. It tests different network configurations and evaluates their
performance on historical stock data.
"""

import numpy as np
import datetime
import json
import os
from itertools import product
import time

import gymnasium as gym
import gym_anytrading

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import yfinance as yf


def get_data_features(ticker, start_date, end_date, interval='1h'):
    """
    Fetch historical stock data for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date for historical data
        end_date: End date for historical data
        interval: Data interval (default: '1h')
    
    Returns:
        DataFrame with historical stock data
    """
    data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
    data.dropna(inplace=True)
    return data


def create_trading_env(df, window_size, frame_bound):
    """
    Create a trading environment using gym_anytrading.
    
    Args:
        df: DataFrame with historical stock data
        window_size: Number of historical data points for prediction
        frame_bound: Tuple defining the range of data to use
    
    Returns:
        Trading environment
    """
    environment = gym.make(
        'stocks-v0',
        df=df,
        window_size=window_size,
        frame_bound=frame_bound
    )
    return environment


def train_and_evaluate_model(env, policy_kwargs, total_timesteps=50000, n_eval_episodes=5):
    """
    Train a PPO model with specific hyperparameters and evaluate its performance.
    
    Args:
        env: Trading environment
        policy_kwargs: Dictionary containing network architecture parameters
        total_timesteps: Number of timesteps to train the model
        n_eval_episodes: Number of episodes to evaluate the model
    
    Returns:
        Dictionary with mean reward, std reward, and training time
    """
    start_time = time.time()
    
    # Create and train the model
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    training_time = time.time() - start_time
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    
    return {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'training_time': float(training_time)
    }


def hyperparameter_search():
    """
    Perform hyperparameter search for model architecture parameters.
    
    Tests different combinations of:
    - Number of layers (net_arch)
    - Number of neurons per layer
    
    Returns:
        List of results for all configurations tested
    """
    print("=" * 80)
    print("PPO Trading Bot - Hyperparameter Search for Model Size Optimization")
    print("=" * 80)
    print()
    
    # Fetch historical data
    ticker = 'AAPL'
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=59)
    
    print(f"Fetching data for {ticker}...")
    df = get_data_features(ticker, start_date, end_date)
    print(f"Data fetched: {len(df)} rows")
    print()
    
    # Environment parameters
    window_size = 50
    frame_bound = (50, len(df) - 1)
    
    # Define hyperparameter search space
    # Testing different network architectures
    network_configs = [
        {'net_arch': [64]},                    # Small: 1 layer, 64 neurons
        {'net_arch': [128]},                   # Small: 1 layer, 128 neurons
        {'net_arch': [64, 64]},                # Medium: 2 layers, 64 neurons each
        {'net_arch': [128, 128]},              # Medium: 2 layers, 128 neurons each
        {'net_arch': [256, 256]},              # Large: 2 layers, 256 neurons each
        {'net_arch': [128, 64]},               # Medium: 2 layers, decreasing
        {'net_arch': [64, 64, 64]},            # Deep: 3 layers, 64 neurons each
        {'net_arch': [128, 128, 64]},          # Deep: 3 layers, decreasing
    ]
    
    # Reduced timesteps for faster search (can be increased for production)
    total_timesteps = 50000
    n_eval_episodes = 5
    
    results = []
    
    print(f"Starting hyperparameter search with {len(network_configs)} configurations...")
    print(f"Training timesteps per config: {total_timesteps:,}")
    print(f"Evaluation episodes: {n_eval_episodes}")
    print()
    print("-" * 80)
    
    for idx, config in enumerate(network_configs, 1):
        print(f"\n[{idx}/{len(network_configs)}] Testing configuration: {config['net_arch']}")
        
        # Create fresh environment for each test
        env = create_trading_env(df, window_size, frame_bound)
        
        try:
            # Train and evaluate
            result = train_and_evaluate_model(env, config, total_timesteps, n_eval_episodes)
            
            # Store configuration and results
            result['config'] = config['net_arch']
            result['total_params'] = sum(config['net_arch']) * 2  # Rough estimate
            results.append(result)
            
            print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  Training Time: {result['training_time']:.2f}s")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'config': config['net_arch'],
                'error': str(e)
            })
        
        finally:
            env.close()
    
    print("\n" + "=" * 80)
    print("Hyperparameter Search Complete!")
    print("=" * 80)
    
    return results


def save_results(results, filename='hyperparameter_search_results.json'):
    """
    Save results to a JSON file.
    
    Args:
        results: List of result dictionaries
        filename: Name of the output file
    """
    output_path = os.path.join(os.path.dirname(__file__), filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def analyze_and_summarize(results):
    """
    Analyze results and print a summary.
    
    Args:
        results: List of result dictionaries
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Filter out failed runs
    successful_results = [r for r in results if 'mean_reward' in r]
    
    if not successful_results:
        print("No successful runs to analyze.")
        return
    
    # Sort by mean reward
    sorted_results = sorted(successful_results, key=lambda x: x['mean_reward'], reverse=True)
    
    print("\nTop 3 Configurations by Mean Reward:")
    print("-" * 80)
    for idx, result in enumerate(sorted_results[:3], 1):
        print(f"\n{idx}. Network Architecture: {result['config']}")
        print(f"   Mean Reward: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        print(f"   Training Time: {result['training_time']:.2f}s")
    
    # Best configuration
    best_config = sorted_results[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Network Architecture: {best_config['config']}")
    print(f"Mean Reward: {best_config['mean_reward']:.4f} ± {best_config['std_reward']:.4f}")
    print(f"Training Time: {best_config['training_time']:.2f}s")
    
    # Performance vs Complexity tradeoff
    print("\n" + "=" * 80)
    print("PERFORMANCE vs COMPLEXITY ANALYSIS")
    print("=" * 80)
    print(f"{'Architecture':<20} {'Mean Reward':<15} {'Training Time':<15} {'Efficiency':<10}")
    print("-" * 80)
    
    for result in sorted_results:
        config_str = str(result['config'])
        efficiency = result['mean_reward'] / result['training_time'] if result['training_time'] > 0 else 0
        print(f"{config_str:<20} {result['mean_reward']:>14.4f} {result['training_time']:>14.2f}s {efficiency:>10.4f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best efficiency (reward per second)
    best_efficiency = max(sorted_results, 
                         key=lambda x: x['mean_reward'] / x['training_time'] if x['training_time'] > 0 else 0)
    
    print(f"\nBest Overall Performance: {best_config['config']}")
    print(f"  → Use this for maximum trading performance")
    print(f"\nBest Efficiency (Reward/Time): {best_efficiency['config']}")
    print(f"  → Use this for faster training cycles")
    
    # Guidance on model size
    if len(best_config['config']) == 1:
        depth = "shallow (1 layer)"
    elif len(best_config['config']) == 2:
        depth = "medium depth (2 layers)"
    else:
        depth = "deep (3+ layers)"
    
    avg_neurons = np.mean(best_config['config'])
    if avg_neurons < 80:
        size = "small"
    elif avg_neurons < 150:
        size = "medium"
    else:
        size = "large"
    
    print(f"\nModel Characteristics: {size} size, {depth}")
    print(f"Total approximate parameters: ~{sum(best_config['config']) * 2}")
    
    print("\n" + "=" * 80)


def main():
    """Main function to run hyperparameter search."""
    # Run the hyperparameter search
    results = hyperparameter_search()
    
    # Save results to file
    save_results(results)
    
    # Analyze and print summary
    analyze_and_summarize(results)
    
    print("\n" + "=" * 80)
    print("Hyperparameter search completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
