# expert_agent.py

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from balatro_gym.balatro_env_2 import BalatroEnv, Phase, Action
from balatro_gym.cards import Card, Suit, Rank
from balatro_gym.scoring_engine import HandType

class BalatroExpertAgent:
    """Expert agent that plays near-optimally for imitation learning"""
    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.shop_evaluator = ShopEvaluator()
        
    def get_action(self, obs: Dict[str, np.ndarray], env: BalatroEnv) -> Tuple[int, Dict[str, Any]]:
        """Get expert action with reasoning"""
        phase = Phase(obs['phase'].item())
        
        if phase == Phase.BLIND_SELECT:
            return self._select_blind(obs, env)
        elif phase == Phase.SHOP:
            return self._shop_decision(obs, env)
        elif phase == Phase.PLAY:
            return self._play_decision(obs, env)
        
        return 0, {}
    
    def _play_decision(self, obs: Dict[str, np.ndarray], env: BalatroEnv) -> Tuple[int, Dict[str, Any]]:
        """Expert play phase decisions"""
        info = {}
        
        # Calculate game state
        progress = obs['round_chips_scored'].item() / max(1, obs['chips_needed'].item())
        hands_left = obs['hands_left'].item()
        chips_needed = obs['chips_needed'].item() - obs['round_chips_scored'].item()
        
        # No cards selected - find best play
        if len(env.state.selected_cards) == 0:
            hand_cards = self._get_hand_cards(obs, env)
            
            # Evaluate all possible hands
            best_play = self.hand_evaluator.find_best_play(
                hand_cards, 
                chips_needed,
                hands_left,
                env.state
            )
            
            info['reasoning'] = best_play.reasoning
            info['expected_value'] = best_play.expected_value
            
            # Select first card of best hand
            if best_play.card_indices:
                return Action.SELECT_CARD_BASE + best_play.card_indices[0], info
        
        # Cards selected - decide whether to play
        else:
            selected_cards = [self._get_hand_cards(obs, env)[i] 
                            for i in env.state.selected_cards]
            hand_value = self.hand_evaluator.evaluate_hand(selected_cards, env.state)
            
            # Check if we should play or keep selecting
            if self._should_play_hand(hand_value, chips_needed, hands_left):
                info['reasoning'] = f"Playing hand worth {hand_value.score} points"
                return Action.PLAY_HAND, info
            
            # Look for cards to add
            next_card = self._find_improving_card(env.state.selected_cards, obs, env)
            if next_card is not None:
                return Action.SELECT_CARD_BASE + next_card, info
            
            # Can't improve, play what we have
            return Action.PLAY_HAND, info
    
    def _should_play_hand(self, hand_value, chips_needed: int, hands_left: int) -> bool:
        """Determine if current hand should be played"""
        if hands_left <= 1:
            return True  # Last hand, must play
        
        # Calculate required average score
        avg_needed = chips_needed / hands_left
        
        # Play if hand exceeds requirement with buffer
        return hand_value.score >= avg_needed * 1.2
    
    def _shop_decision(self, obs: Dict[str, np.ndarray], env: BalatroEnv) -> Tuple[int, Dict[str, Any]]:
        """Expert shop decisions"""
        if not env.shop:
            return Action.SHOP_END, {'reasoning': 'No shop available'}
        
        money = obs['money'].item()
        evaluations = []
        
        # Evaluate each item
        for i, item in enumerate(env.shop.inventory):
            if money >= item.cost:
                value = self.shop_evaluator.evaluate_purchase(item, env.state)
                evaluations.append((i, value, item))
        
        # Sort by value
        evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # Buy best item if value is positive
        if evaluations and evaluations[0][1] > 0:
            idx, value, item = evaluations[0]
            return Action.SHOP_BUY_BASE + idx, {
                'reasoning': f"Buying {item.name} (value: {value:.1f})"
            }
        
        # Consider reroll if we have money
        if money >= obs['shop_rerolls'].item() and money > 50:
            reroll_value = self.shop_evaluator.evaluate_reroll(env.shop.inventory, money)
            if reroll_value > 0:
                return Action.SHOP_REROLL, {'reasoning': 'Rerolling for better items'}
        
        return Action.SHOP_END, {'reasoning': 'Nothing worth buying'}

class HandEvaluator:
    """Evaluates poker hands and finds optimal plays"""
    
    def find_best_play(self, hand: List[Card], chips_needed: int, 
                      hands_left: int, game_state) -> 'PlayDecision':
        """Find the best cards to play from hand"""
        candidates = []
        
        # Try all combinations up to 5 cards
        from itertools import combinations
        
        for r in range(1, 6):
            for combo in combinations(range(len(hand)), r):
                if all(hand[i] for i in combo):
                    cards = [hand[i] for i in combo]
                    value = self.evaluate_hand(cards, game_state)
                    candidates.append(PlayDecision(combo, value, cards))
        
        # Sort by expected value
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        
        if candidates:
            best = candidates[0]
            best.reasoning = f"Best hand: {best.hand_type} worth ~{best.expected_value:.0f}"
            return best
        
        return PlayDecision([], HandValue(0, HandType.HIGH_CARD), [])
    
    def evaluate_hand(self, cards: List[Card], game_state) -> 'HandValue':
        """Evaluate a poker hand's value"""
        # Simplified - would use actual scoring engine
        hand_type = self._classify_hand(cards)
        base_score = len(cards) * 10  # Placeholder
        
        return HandValue(base_score, hand_type)
    
    def _classify_hand(self, cards: List[Card]) -> HandType:
        """Classify poker hand type"""
        # Simplified classification
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        
        for card in cards:
            rank_counts[card.rank] += 1
            suit_counts[card.suit] += 1
        
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        if max_rank_count >= 4:
            return HandType.FOUR_KIND
        elif max_rank_count >= 3:
            return HandType.THREE_KIND
        elif max_rank_count >= 2:
            return HandType.ONE_PAIR
        elif max_suit_count >= 5:
            return HandType.FLUSH
        else:
            return HandType.HIGH_CARD

class PlayDecision:
    def __init__(self, indices, value, cards):
        self.card_indices = indices
        self.expected_value = value.score if hasattr(value, 'score') else 0
        self.hand_type = value.hand_type if hasattr(value, 'hand_type') else HandType.HIGH_CARD
        self.cards = cards
        self.reasoning = ""

class HandValue:
    def __init__(self, score, hand_type):
        self.score = score
        self.hand_type = hand_type

class ShopEvaluator:
    """Evaluates shop purchases"""
    
    def evaluate_purchase(self, item, game_state) -> float:
        """Evaluate the value of purchasing an item"""
        # Simplified evaluation
        if "Joker" in item.name:
            # Jokers are valuable early
            value = 50.0 - game_state.ante * 5
            if len(game_state.jokers) == 0:
                value *= 2  # First joker is crucial
            return value
        
        return 10.0  # Default value
    
    def evaluate_reroll(self, current_items, money) -> float:
        """Evaluate whether to reroll shop"""
        # Reroll if we have money and bad items
        if money > 100:
            return 20.0
        return -10.0
