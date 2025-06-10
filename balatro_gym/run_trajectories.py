"""run_balatro_trajectories.py - Generate trajectories from the Balatro environment"""

import numpy as np
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import datetime

# Import your environment
from balatro_env_v2 import BalatroEnvComplete

@dataclass
class Transition:
    """Single transition in a trajectory"""
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    
@dataclass
class Trajectory:
    """Complete trajectory for one episode"""
    transitions: List[Transition]
    total_reward: float
    final_chips: int
    final_ante: int
    hands_played: int
    jokers_acquired: List[str]
    
class TrajectoryCollector:
    """Collects trajectories from Balatro environment"""
    
    def __init__(self, env: BalatroEnvComplete):
        self.env = env
        self.reset_stats()
        
    def reset_stats(self):
        """Reset collection statistics"""
        self.stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'total_reward': 0,
            'max_ante_reached': 0,
            'max_chips': 0,
            'hand_types_played': {},
            'jokers_purchased': {},
        }
    
    def collect_trajectory(self, policy='random', max_steps=1000, verbose=False) -> Trajectory:
        """Collect a single trajectory"""
        transitions = []
        total_reward = 0
        hands_played = 0
        jokers_acquired = []
        
        # Reset environment
        obs, info = self.env.reset()
        
        for step in range(max_steps):
            # Get action from policy
            if policy == 'random':
                action = self._random_policy(obs)
            elif policy == 'smart':
                action = self._smart_policy(obs)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                info=info
            )
            transitions.append(transition)
            
            # Update stats
            total_reward += reward
            if 'hand_type' in info:
                hands_played += 1
                hand_type = info['hand_type']
                self.stats['hand_types_played'][hand_type] = self.stats['hand_types_played'].get(hand_type, 0) + 1
            
            # Track joker purchases
            if obs['phase'] == 1 and 'inventory' in obs:  # Shop phase
                # Check if we bought a joker
                if action >= 12 and action < 20:  # Buy joker range
                    idx = action - 12
                    if idx < len(self.env.shop.inventory):
                        item = self.env.shop.inventory[idx]
                        if hasattr(item, 'name') and 'Joker' in str(item.name):
                            jokers_acquired.append(item.name)
                            self.stats['jokers_purchased'][item.name] = self.stats['jokers_purchased'].get(item.name, 0) + 1
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: Action {action}, Reward {reward:.2f}, Chips: {next_obs['chips']}")
            
            # Render if needed
            if verbose:
                self.env.render()
            
            obs = next_obs
            
            if done:
                break
        
        # Create trajectory
        trajectory = Trajectory(
            transitions=transitions,
            total_reward=total_reward,
            final_chips=int(obs['chips']),
            final_ante=int(obs['ante']),
            hands_played=hands_played,
            jokers_acquired=jokers_acquired
        )
        
        # Update global stats
        self.stats['total_episodes'] += 1
        self.stats['total_steps'] += len(transitions)
        self.stats['total_reward'] += total_reward
        self.stats['max_ante_reached'] = max(self.stats['max_ante_reached'], trajectory.final_ante)
        self.stats['max_chips'] = max(self.stats['max_chips'], trajectory.final_chips)
        
        return trajectory
    
    def _random_policy(self, obs: Dict) -> int:
        """Random valid action selection"""
        action_mask = obs['action_mask']
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)
    
    def _smart_policy(self, obs: Dict) -> int:
        """Smarter policy that makes reasonable decisions"""
        action_mask = obs['action_mask']
        valid_actions = np.where(action_mask)[0]
        
        if obs['phase'] == 0:  # Play phase
            # Prefer playing hands over discarding
            if obs.get('has_flush', 0):
                return 5  # Play flush
            elif obs.get('has_three_kind', 0):
                return 4  # Play three of a kind
            elif obs.get('has_two_pair', 0):
                return 3  # Play two pair
            elif obs.get('has_pair', 0):
                return 2  # Play pair
            elif 0 in valid_actions:
                return 0  # Play best hand
            elif 1 in valid_actions:
                return 1  # Play first 5
            else:
                # Discard if we must
                if 6 in valid_actions and obs['discards_left'] > 0:
                    return 6  # Discard 1
        
        else:  # Shop phase
            # Buy jokers if we can afford them
            if 'shop_cost' in obs:
                costs = obs['shop_cost']
                chips = obs['chips']
                
                # Try to buy jokers (indices 0-2 typically)
                for i in range(min(3, len(costs))):
                    if costs[i] <= chips and (12 + i) in valid_actions:
                        return 12 + i  # Buy joker
                
                # Skip if nothing good to buy
                if 10 in valid_actions:  # Skip
                    return 10
        
        # Fallback to random
        return np.random.choice(valid_actions)
    
    def collect_multiple_trajectories(self, 
                                    n_trajectories: int, 
                                    policy: str = 'random',
                                    max_steps_per_episode: int = 1000,
                                    verbose: bool = False) -> List[Trajectory]:
        """Collect multiple trajectories"""
        trajectories = []
        
        print(f"Collecting {n_trajectories} trajectories with {policy} policy...")
        
        for i in range(n_trajectories):
            if verbose or (i + 1) % 10 == 0:
                print(f"\nTrajectory {i + 1}/{n_trajectories}")
            
            trajectory = self.collect_trajectory(
                policy=policy,
                max_steps=max_steps_per_episode,
                verbose=verbose and i < 3  # Only verbose for first 3
            )
            
            trajectories.append(trajectory)
            
            # Print summary
            print(f"  Completed: {len(trajectory.transitions)} steps, "
                  f"Reward: {trajectory.total_reward:.1f}, "
                  f"Final: Ante {trajectory.final_ante}, ${trajectory.final_chips}")
        
        return trajectories
    
    def save_trajectories(self, trajectories: List[Trajectory], filename: str):
        """Save trajectories to file"""
        # Convert to serializable format
        data = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'n_trajectories': len(trajectories),
                'env': 'BalatroEnvComplete',
                'stats': self.stats
            },
            'trajectories': []
        }
        
        for traj in trajectories:
            traj_data = {
                'total_reward': traj.total_reward,
                'final_chips': traj.final_chips,
                'final_ante': traj.final_ante,
                'hands_played': traj.hands_played,
                'jokers_acquired': traj.jokers_acquired,
                'transitions': []
            }
            
            # Convert transitions
            for trans in traj.transitions:
                trans_data = {
                    'state': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in trans.state.items()},
                    'action': int(trans.action),
                    'reward': float(trans.reward),
                    'next_state': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in trans.next_state.items()},
                    'done': bool(trans.done),
                    'info': trans.info
                }
                traj_data['transitions'].append(trans_data)
            
            data['trajectories'].append(traj_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSaved {len(trajectories)} trajectories to {filename}")
    
    def print_statistics(self):
        """Print collection statistics"""
        print("\n=== COLLECTION STATISTICS ===")
        print(f"Total episodes: {self.stats['total_episodes']}")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Average steps per episode: {self.stats['total_steps'] / max(1, self.stats['total_episodes']):.1f}")
        print(f"Total reward: {self.stats['total_reward']:.1f}")
        print(f"Average reward per episode: {self.stats['total_reward'] / max(1, self.stats['total_episodes']):.1f}")
        print(f"Max ante reached: {self.stats['max_ante_reached']}")
        print(f"Max chips: ${self.stats['max_chips']}")
        
        print("\nHand types played:")
        for hand_type, count in sorted(self.stats['hand_types_played'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {hand_type}: {count}")
        
        if self.stats['jokers_purchased']:
            print("\nJokers purchased:")
            for joker, count in sorted(self.stats['jokers_purchased'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {joker}: {count}")

def main():
    """Main trajectory collection script"""
    # Create environment
    env = BalatroEnvComplete(render_mode="human")
    collector = TrajectoryCollector(env)
    
    # Collect trajectories with random policy
    print("=== RANDOM POLICY ===")
    random_trajectories = collector.collect_multiple_trajectories(
        n_trajectories=50,
        policy='random',
        max_steps_per_episode=500,
        verbose=False
    )
    
    # Save random trajectories
    collector.save_trajectories(random_trajectories, 'balatro_trajectories_random.json')
    collector.print_statistics()
    
    # Reset stats and collect with smart policy
    print("\n\n=== SMART POLICY ===")
    collector.reset_stats()
    smart_trajectories = collector.collect_multiple_trajectories(
        n_trajectories=50,
        policy='smart',
        max_steps_per_episode=500,
        verbose=False
    )
    
    # Save smart trajectories
    collector.save_trajectories(smart_trajectories, 'balatro_trajectories_smart.json')
    collector.print_statistics()
    
    # Analyze trajectory quality
    print("\n\n=== TRAJECTORY ANALYSIS ===")
    
    def analyze_trajectories(trajectories: List[Trajectory], name: str):
        print(f"\n{name} Policy Analysis:")
        rewards = [t.total_reward for t in trajectories]
        antes = [t.final_ante for t in trajectories]
        chips = [t.final_chips for t in trajectories]
        steps = [len(t.transitions) for t in trajectories]
        
        print(f"  Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg final ante: {np.mean(antes):.2f} ± {np.std(antes):.2f}")
        print(f"  Avg final chips: ${np.mean(chips):.0f} ± ${np.std(chips):.0f}")
        print(f"  Avg episode length: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  Max reward: {max(rewards):.2f}")
        print(f"  Max ante reached: {max(antes)}")
        print(f"  Max chips: ${max(chips)}")
    
    analyze_trajectories(random_trajectories, "Random")
    analyze_trajectories(smart_trajectories, "Smart")
    
    env.close()

if __name__ == "__main__":
    main()
