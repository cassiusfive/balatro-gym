# balatro_gym/trajectory_analysis.py
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class TrajectoryAnalyzer:
    """Analyze generated trajectories for quality and coverage"""
    
    def __init__(self, trajectory_dir: str):
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectories = self._load_all_trajectories()
        
    def analyze_performance(self):
        """Analyze expert performance across different joker combinations"""
        
        results = defaultdict(list)
        
        for traj in self.trajectories:
            joker_key = tuple(sorted(traj['metadata']['jokers']))
            results['joker_combination'].append(joker_key)
            results['total_reward'].append(traj['metadata']['total_reward'])
            results['ante_reached'].append(traj['metadata']['ante_reached'])
            results['trajectory_length'].append(traj['metadata']['length'])
            
        df = pd.DataFrame(results)
        
        # Performance by joker combination
        print("Performance by Joker Combination:")
        print(df.groupby('joker_combination').agg({
            'total_reward': ['mean', 'std', 'max'],
            'ante_reached': ['mean', 'max'],
            'trajectory_length': 'mean'
        }))
        
        return df
    
    def analyze_decisions(self):
        """Analyze expert decision patterns"""
        
        decision_stats = {
            'play_vs_discard': defaultdict(int),
            'target_hands': defaultdict(int),
            'discard_counts': defaultdict(int)
        }
        
        for traj in self.trajectories:
            for step in traj['steps']:
                reasoning = step.get('expert_reasoning', {})
                
                # Play vs discard decisions
                decision_type = reasoning.get('decision_type', 'unknown')
                decision_stats['play_vs_discard'][decision_type] += 1
                
                # Target hands when discarding
                if decision_type == 'discard':
                    target = reasoning.get('target_hand', 'unknown')
                    decision_stats['target_hands'][target] += 1
                    
        return decision_stats
    
    def plot_learning_curves(self):
        """Plot performance over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rewards over episodes
        rewards = [t['metadata']['total_reward'] for t in self.trajectories]
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Ante reached
        antes = [t['metadata']['ante_reached'] for t in self.trajectories]
        axes[0, 1].plot(antes)
        axes[0, 1].set_title('Ante Reached per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Ante')
        
        # Decision distribution
        decisions = self.analyze_decisions()
        decision_types = list(decisions['play_vs_discard'].keys())
        decision_counts = list(decisions['play_vs_discard'].values())
        axes[1, 0].bar(decision_types, decision_counts)
        axes[1, 0].set_title('Decision Type Distribution')
        
        # Target hand distribution
        target_hands = list(decisions['target_hands'].keys())
        target_counts = list(decisions['target_hands'].values())
        axes[1, 1].bar(range(len(target_hands)), target_counts)
        axes[1, 1].set_xticks(range(len(target_hands)))
        axes[1, 1].set_xticklabels(target_hands, rotation=45)
        axes[1, 1].set_title('Target Hand Distribution')
        
        plt.tight_layout()
        plt.savefig('trajectory_analysis.png')
        plt.show()
