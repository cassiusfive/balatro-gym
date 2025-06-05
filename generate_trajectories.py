# generate_expert_trajectories.py
import argparse
from balatro_gym.trajectory_generator import TrajectoryGenerator
from balatro_gym.trajectory_analysis import TrajectoryAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='expert_trajectories')
    parser.add_argument('--analyze_only', action='store_true')
    args = parser.parse_args()
    
    if not args.analyze_only:
        print(f"Generating {args.num_trajectories} expert trajectories...")
        
        generator = TrajectoryGenerator(save_dir=args.save_dir)
        
        # Generate trajectories with curriculum
        trajectories = generator.generate_trajectories(
            num_trajectories=args.num_trajectories,
            save_every=1000
        )
        
        print(f"Generated {len(trajectories)} trajectories")
        
    # Analyze results
    print("\nAnalyzing trajectories...")
    analyzer = TrajectoryAnalyzer(args.save_dir)
    
    df = analyzer.analyze_performance()
    analyzer.plot_learning_curves()
    
    print("\nTrajectory generation complete!")
    print(f"Average reward: {df['total_reward'].mean():.2f}")
    print(f"Average ante reached: {df['ante_reached'].mean():.2f}")
    
if __name__ == "__main__":
    main()# generate_expert_trajectories.py
import argparse
from balatro_gym.trajectory_generator import TrajectoryGenerator
from balatro_gym.trajectory_analysis import TrajectoryAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='expert_trajectories')
    parser.add_argument('--analyze_only', action='store_true')
    args = parser.parse_args()
    
    if not args.analyze_only:
        print(f"Generating {args.num_trajectories} expert trajectories...")
        
        generator = TrajectoryGenerator(save_dir=args.save_dir)
        
        # Generate trajectories with curriculum
        trajectories = generator.generate_trajectories(
            num_trajectories=args.num_trajectories,
            save_every=1000
        )
        
        print(f"Generated {len(trajectories)} trajectories")
        
    # Analyze results
    print("\nAnalyzing trajectories...")
    analyzer = TrajectoryAnalyzer(args.save_dir)
    
    df = analyzer.analyze_performance()
    analyzer.plot_learning_curves()
    
    print("\nTrajectory generation complete!")
    print(f"Average reward: {df['total_reward'].mean():.2f}")
    print(f"Average ante reached: {df['ante_reached'].mean():.2f}")
    
if __name__ == "__main__":
    main()
