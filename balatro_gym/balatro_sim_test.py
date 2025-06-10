import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import json

@dataclass
class Card:
    rank: int  # 2-14 (11=J, 12=Q, 13=K, 14=A)
    suit: str  # Hearts, Diamonds, Clubs, Spades
    enhancement: str = None  # bonus, mult, wild, glass, steel, stone, gold, lucky
    edition: str = None  # foil, holographic, polychrome, negative
    seal: str = None  # red, blue, gold, purple
    base_value: int = None
    
    def __post_init__(self):
        if self.base_value is None:
            # Standard card values (A=11, face cards=10, rest=face value)
            if self.rank == 14:  # Ace
                self.base_value = 11
            elif self.rank >= 11:  # Face cards
                self.base_value = 10
            else:
                self.base_value = self.rank

@dataclass
class Joker:
    name: str
    type: str = "Joker"
    sell_value: int = 2
    
@dataclass
class GameState:
    """Complete Balatro game state"""
    money: int = 4
    ante: int = 1
    hands_left: int = 4
    discards_left: int = 3
    hand: List[Card] = None
    deck: List[Card] = None
    jokers: List[Joker] = None
    consumables: List[str] = None
    hand_levels: Dict[str, int] = None
    score: int = 0
    blind_requirement: int = 300
    
    def __post_init__(self):
        if self.hand is None:
            self.hand = []
        if self.deck is None:
            self.deck = self.create_standard_deck()
        if self.jokers is None:
            self.jokers = []
        if self.consumables is None:
            self.consumables = []
        if self.hand_levels is None:
            self.hand_levels = {hand: 1 for hand in [
                'High Card', 'Pair', 'Two Pair', 'Three of a Kind', 
                'Straight', 'Flush', 'Full House', 'Four of a Kind',
                'Straight Flush', 'Five of a Kind', 'Flush House', 'Flush Five'
            ]}
    
    def create_standard_deck(self):
        """Create standard 52-card deck"""
        deck = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        for suit in suits:
            for rank in range(2, 15):  # 2-14 (A=14)
                deck.append(Card(rank=rank, suit=suit))
        return deck
    
    def draw_hand(self, size=8):
        """Draw cards to hand"""
        drawn = []
        for _ in range(min(size, len(self.deck))):
            if self.deck:
                drawn.append(self.deck.pop())
        self.hand.extend(drawn)
        return drawn

class BalatroTrajectoryGenerator:
    def __init__(self, simulator):
        self.simulator = simulator
        
    def test_basic_scoring(self):
        """Test basic hand evaluation and scoring"""
        print("=== TESTING BASIC SCORING ===")
        
    
    # Test 1: High Card
        cards = [
        Card(14, 'Hearts'),  # Ace
        Card(12, 'Diamonds'),  # Queen  
        Card(9, 'Clubs'),
        Card(6, 'Spades'),
        Card(3, 'Hearts')
        ]
    
        print(f"Testing with cards: {[(c.rank, c.suit) for c in cards]}")
    
        result = self.simulator.evaluate_hand(cards)
        print(f"Result returned: {result}")
        print(f"Result type: {type(result)}")
    
        if result is None:
           print("ERROR: evaluate_hand returned None!")
           return
        
        if 'top' not in result:
           print(f"ERROR: No 'top' key in result. Keys: {list(result.keys())}")
           return
        
        print(f"Hand: A♥ Q♦ 9♣ 6♠ 3♥")
        print(f"Detected: {result['top']}")
        print(f"Expected: High Card")
        print() 
        # Test 2: Pair
        cards = [
            Card(9, 'Hearts'),
            Card(9, 'Diamonds'),
            Card(12, 'Clubs'), 
            Card(6, 'Spades'),
            Card(3, 'Hearts')
        ]
        
        result = self.simulator.evaluate_hand(cards)
        print(f"Hand: 9♥ 9♦ Q♣ 6♠ 3♥")
        print(f"Detected: {result['top']}")
        print(f"Expected: Pair")
        print()
        
        # Test 3: Flush
        cards = [
            Card(14, 'Hearts'),
            Card(10, 'Hearts'),
            Card(7, 'Hearts'),
            Card(5, 'Hearts'),
            Card(2, 'Hearts')
        ]
        
        result = self.simulator.evaluate_hand(cards)
        print(f"Hand: A♥ 10♥ 7♥ 5♥ 2♥")
        print(f"Detected: {result['top']}")
        print(f"Expected: Flush")
        print()
        
        # Test 4: Straight
        cards = [
            Card(10, 'Hearts'),
            Card(9, 'Diamonds'),
            Card(8, 'Clubs'),
            Card(7, 'Spades'),
            Card(6, 'Hearts')
        ]
        
        result = self.simulator.evaluate_hand(cards)
        print(f"Hand: 10♥ 9♦ 8♣ 7♠ 6♥")
        print(f"Detected: {result['top']}")
        print(f"Expected: Straight")
        print()
    
    def test_joker_effects(self):
        """Test individual joker effects"""
        print("=== TESTING JOKER EFFECTS ===")
        
        # Test Fibonacci joker
        cards = [
            Card(5, 'Hearts'),   # Fibonacci number
            Card(8, 'Diamonds'), # Fibonacci number
            Card(10, 'Clubs'),
            Card(6, 'Spades'),
            Card(3, 'Hearts')    # Fibonacci number
        ]
        
        jokers = [Joker('Fibonacci')]
        game_state = GameState()
        
        score, _ = self.simulator.calculate_score(cards, jokers, game_state.hand_levels, game_state.__dict__)
        print(f"Hand with Fibonacci joker: {score}")
        print(f"Cards: 5♥ 8♦ 10♣ 6♠ 3♥ (3 Fibonacci numbers)")
        print()
        
        # Test Scholar joker
        cards = [
            Card(14, 'Hearts'),  # Ace for Scholar
            Card(12, 'Diamonds'),
            Card(9, 'Clubs'),
            Card(6, 'Spades'),
            Card(3, 'Hearts')
        ]
        
        jokers = [Joker('Scholar')]
        score, _ = self.simulator.calculate_score(cards, jokers, game_state.hand_levels, game_state.__dict__)
        print(f"High Card with Scholar joker (Ace): {score}")
        print()
    
    def test_enhanced_cards(self):
        """Test card enhancements"""
        print("=== TESTING ENHANCED CARDS ===")
        
        # Test steel card
        cards = [
            Card(14, 'Hearts', enhancement='steel'),  # Steel Ace
            Card(12, 'Diamonds'),
            Card(9, 'Clubs'),
            Card(6, 'Spades'),
            Card(3, 'Hearts')
        ]
        
        game_state = GameState()
        score, _ = self.simulator.calculate_score(cards, [], game_state.hand_levels, game_state.__dict__)
        print(f"High Card with Steel Ace: {score}")
        print()
        
        # Test glass card (risky!)
        cards[0].enhancement = 'glass'
        score, _ = self.simulator.calculate_score(cards, [], game_state.hand_levels, game_state.__dict__)
        print(f"High Card with Glass Ace: {score}")
        print()
    
    def test_complete_game_state(self):
        """Test with complete game state"""
        print("=== TESTING COMPLETE GAME STATE ===")
        
        # Create realistic game state
        game_state = GameState(
            money=15,
            ante=2, 
            hands_left=3,
            discards_left=2
        )
        
        # Add some jokers
        game_state.jokers = [
            Joker('Joker'),  # +4 mult
            Joker('Even Steven'),  # +4 mult for even cards
            Joker('Blue Joker')  # +2 chips per remaining deck card
        ]
        
        # Create hand
        cards = [
            Card(8, 'Hearts'),   # Even for Even Steven
            Card(6, 'Diamonds'), # Even for Even Steven  
            Card(12, 'Clubs'),   # Face card
            Card(4, 'Spades'),   # Even for Even Steven
            Card(2, 'Hearts')    # Even for Even Steven
        ]
        
        # Upgrade some hands with planets
        game_state.hand_levels['Pair'] = 3  # Upgraded with planets
        
        score, updated_state = self.simulator.calculate_score(
            cards, game_state.jokers, game_state.hand_levels, game_state.__dict__
        )
        
        print(f"Complete game score: {score}")
        print(f"Jokers: {[j.name for j in game_state.jokers]}")
        print(f"Hand levels: Pair level {game_state.hand_levels['Pair']}")
        print(f"Deck size: {len(game_state.deck)} cards")
        print()
    
    def generate_single_trajectory(self, max_actions=50):
        """Generate a single game trajectory for RL"""
        print("=== GENERATING RL TRAJECTORY ===")
        
        trajectory = []
        game_state = GameState()
        game_state.draw_hand(8)
        
        action_count = 0
        
        while action_count < max_actions and game_state.hands_left > 0:
            # Current state observation
            state_obs = self.state_to_observation(game_state)
            
            # Sample random action (replace with RL agent later)
            available_actions = self.get_available_actions(game_state)
            action = random.choice(available_actions)
            
            # Execute action
            reward, next_state = self.execute_action(action, game_state)
            
            # Next state observation
            next_obs = self.state_to_observation(next_state)
            
            # Store transition
            transition = {
                'state': state_obs,
                'action': action,
                'reward': reward,
                'next_state': next_obs,
                'done': next_state.hands_left <= 0
            }
            
            trajectory.append(transition)
            
            print(f"Action {action_count}: {action['type']}")
            print(f"  Reward: {reward}")
            print(f"  Score: {next_state.score}")
            print(f"  Money: {next_state.money}")
            print(f"  Hands left: {next_state.hands_left}")
            print()
            
            game_state = next_state
            action_count += 1
            
            if next_state.hands_left <= 0:
                break
        
        print(f"Trajectory complete: {len(trajectory)} transitions")
        print(f"Final score: {game_state.score}")
        print(f"Final money: {game_state.money}")
        
        return trajectory
    
    def state_to_observation(self, game_state):
        """Convert game state to RL observation"""
        obs = {
            'money': game_state.money,
            'ante': game_state.ante,
            'hands_left': game_state.hands_left,
            'discards_left': game_state.discards_left,
            'score': game_state.score,
            'blind_requirement': game_state.blind_requirement,
            
            # Hand cards (rank, suit, enhancement encoding)
            'hand_cards': [(c.rank, c.suit, c.enhancement or 'none') for c in game_state.hand],
            
            # Jokers
            'jokers': [j.name for j in game_state.jokers],
            
            # Hand levels
            'hand_levels': game_state.hand_levels.copy(),
            
            # Deck size
            'deck_size': len(game_state.deck),
            
            # Consumables
            'consumables': len(game_state.consumables)
        }
        
        return obs
    
    def get_available_actions(self, game_state):
        """Get valid actions for current state"""
        actions = []
        
        if len(game_state.hand) >= 1:
            # Can play 1-5 cards
            for num_cards in range(1, min(6, len(game_state.hand) + 1)):
                actions.append({
                    'type': 'play_hand',
                    'card_indices': list(range(num_cards))
                })
        
        if len(game_state.hand) >= 1 and game_state.discards_left > 0:
            # Can discard 1-5 cards
            for num_cards in range(1, min(6, len(game_state.hand) + 1)):
                actions.append({
                    'type': 'discard',
                    'card_indices': list(range(num_cards))
                })
        
        # Use consumables
        for i, consumable in enumerate(game_state.consumables):
            actions.append({
                'type': 'use_consumable',
                'consumable_index': i
            })
        
        return actions
    
    def execute_action(self, action, game_state):
        """Execute action and return reward + next state"""
        import copy
        next_state = copy.deepcopy(game_state)
        reward = 0
        
        if action['type'] == 'play_hand':
            # Play selected cards
            played_cards = [next_state.hand[i] for i in action['card_indices']]
            
            # Calculate score
            score, updated_game_state = self.simulator.calculate_score(
                played_cards, next_state.jokers, next_state.hand_levels, next_state.__dict__
            )
            
            next_state.score += score
            next_state.money = updated_game_state.get('money', next_state.money)
            
            # Remove played cards
            for i in sorted(action['card_indices'], reverse=True):
                next_state.hand.pop(i)
            
            # Reward based on score
            reward = score / 100  # Scale score to reasonable reward
            
            # Check if blind beaten
            if next_state.score >= next_state.blind_requirement:
                reward += 100  # Big bonus for beating blind
                next_state.hands_left = 0  # End round
            else:
                next_state.hands_left -= 1
                
        elif action['type'] == 'discard':
            # Discard selected cards
            for i in sorted(action['card_indices'], reverse=True):
                next_state.hand.pop(i)
            
            next_state.discards_left -= 1
            reward = -1  # Small penalty for discarding
            
        elif action['type'] == 'use_consumable':
            # Use consumable (simplified)
            consumable = next_state.consumables.pop(action['consumable_index'])
            reward = 5  # Reward for using consumable
        
        # Draw back to hand size if needed
        while len(next_state.hand) < 8 and next_state.deck:
            next_state.hand.append(next_state.deck.pop())
        
        return reward, next_state
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("BALATRO SIMULATOR TEST SUITE")
        print("=" * 50)
        
        self.test_basic_scoring()
        self.test_joker_effects()
        self.test_enhanced_cards()
        self.test_complete_game_state()
        self.generate_single_trajectory()
        
        print("=" * 50)
        print("ALL TESTS COMPLETE")

# Usage example
if __name__ == "__main__":
    # Import your simulator
    from balatro_sim import BalatroSimulator  # Adjust import as needed
    
    # Create simulator and test framework
    simulator = BalatroSimulator()
    tester = BalatroTrajectoryGenerator(simulator)
    
    # Run all tests
    tester.run_all_tests()
    
    # Generate multiple trajectories for training
    print("\n=== GENERATING TRAINING TRAJECTORIES ===")
    trajectories = []
    for i in range(5):
        print(f"\nGenerating trajectory {i+1}...")
        trajectory = tester.generate_single_trajectory(max_actions=20)
        trajectories.append(trajectory)
    
    print(f"\nGenerated {len(trajectories)} trajectories")
    print(f"Total transitions: {sum(len(t) for t in trajectories)}")
    
    # Save trajectories for training
    with open('balatro_trajectories.json', 'w') as f:
        json.dump(trajectories, f, indent=2, default=str)
    
    print("Trajectories saved to balatro_trajectories.json")
