# balatro_gym/trajectory_generator.py
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

class TrajectoryGenerator:
    """Generate expert trajectories for imitation learning"""
    
    def __init__(self, env_name='BalatroEnv-v2', save_dir='trajectories'):
        self.env = gym.make(env_name, use_expert_for_actions=True)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def generate_trajectories(self, 
                            num_trajectories: int,
                            joker_configs: Optional[List[List[str]]] = None,
                            save_every: int = 100):
        """Generate expert trajectories with various joker combinations"""
        
        if joker_configs is None:
            # Default joker combinations for curriculum
            joker_configs = self._get_curriculum_jokers()
            
        all_trajectories = []
        
        with tqdm(total=num_trajectories) as pbar:
            for i in range(num_trajectories):
                # Select joker configuration
                jokers = joker_configs[i % len(joker_configs)]
                
                # Generate trajectory
                trajectory = self.generate_single_trajectory(jokers)
                
                # Add metadata
                trajectory['metadata'] = {
                    'trajectory_id': i,
                    'jokers': jokers,
                    'total_reward': sum(step['reward'] for step in trajectory['steps']),
                    'length': len(trajectory['steps']),
                    'ante_reached': trajectory['final_ante']
                }
                
                all_trajectories.append(trajectory)
                pbar.update(1)
                
                # Save periodically
                if (i + 1) % save_every == 0:
                    self._save_trajectories(all_trajectories, f'batch_{i//save_every}')
                    
        # Save final batch
        self._save_trajectories(all_trajectories, 'final')
        
        return all_trajectories
    
    def generate_single_trajectory(self, jokers: List[str]) -> Dict:
        """Generate one expert trajectory"""
        
        # Set jokers for this run
        self.env.set_jokers(jokers)
        
        obs = self.env.reset()
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'expert_reasoning': [],
            'steps': []
        }
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Max steps safety
            # Get current state
            state = self.env.get_full_state()
            
            # Expert decision with reasoning
            expert_decision = self.env.expert.decide_play(
                self.env._state_to_cards(state['hand']),
                jokers,
                state
            )
            
            # Get gym action
            action = self.env._decision_to_action(expert_decision)
            
            # Store pre-step info
            trajectory['states'].append(obs)
            trajectory['actions'].append(action)
            trajectory['expert_reasoning'].append({
                'decision_type': expert_decision['action'],
                'target_hand': expert_decision.get('target_hand', None),
                'expected_score': expert_decision.get('expected_score', 0)
            })
            
            # Execute step
            obs, reward, done, info = self.env.step(action)
            
            # Store post-step info
            trajectory['rewards'].append(reward)
            trajectory['steps'].append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': self.env.get_full_state() if not done else None,
                'done': done,
                'info': info
            })
            
            step_count += 1
            
        trajectory['final_ante'] = self.env.current_ante
        trajectory['total_score'] = self.env.total_score
        
        return trajectory
    
    def _get_curriculum_jokers(self) -> List[List[str]]:
        """Get joker combinations for curriculum learning"""
        return [
            # Stage 1: No jokers
            [],
            
            # Stage 2: Single simple jokers
            ["Joker"],
            ["Greedy Joker"],
            ["Lusty Joker"],
            
            # Stage 3: Conditional jokers
            ["Even Steven"],
            ["Odd Todd"],
            ["Fibonacci"],
            
            # Stage 4: Pairs
            ["Joker", "Greedy Joker"],
            ["Baron", "Mime"],
            ["Even Steven", "Scholar"],
            
            # Stage 5: Complex combinations
            ["Four Fingers", "Flush Five"],
            ["Baron", "Mime", "Joker"],
            ["Fibonacci", "Shortcut", "Joker"],
        ]
    
    def _save_trajectories(self, trajectories: List[Dict], batch_name: str):
        """Save trajectories to disk"""
        
        # Save as pickle for full data
        with open(self.save_dir / f'{batch_name}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)
            
        # Save summary as JSON for analysis
        summary = []
        for traj in trajectories:
            summary.append({
                'id': traj['metadata']['trajectory_id'],
                'jokers': traj['metadata']['jokers'],
                'total_reward': traj['metadata']['total_reward'],
                'length': traj['metadata']['length'],
                'ante_reached': traj['metadata']['ante_reached']
            })
            
        with open(self.save_dir / f'{batch_name}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
