"""balatro_gym/complete_joker_effects.py - Simplified but functional joker effects"""

from typing import Dict, Any, Optional, List
import random

class CompleteJokerEffects:
    """Simplified implementation of joker effects that covers the most common jokers"""
    
    def __init__(self):
        self.joker_states = {}
    
    def apply_joker_effect(self, joker: Any, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a joker's effect based on context"""
        
        if not hasattr(joker, 'name'):
            return None
            
        joker_name = joker.name
        phase = context.get('phase', '')
        
        # Route to appropriate effect based on phase
        if phase == 'scoring':
            return self._scoring_effects(joker_name, context, game_state)
        elif phase == 'individual_scoring':
            return self._individual_scoring_effects(joker_name, context, game_state)
        elif phase == 'discard':
            return self._discard_effects(joker_name, context, game_state)
        elif phase == 'before_scoring':
            return self._before_scoring_effects(joker_name, context, game_state)
        elif phase == 'skip_blind':
            return self._skip_blind_effects(joker_name, context, game_state)
        
        return None
    
    def _scoring_effects(self, joker_name: str, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effects that apply during main scoring phase"""
        
        # Basic mult/chip jokers
        basic_effects = {
            'Joker': {'mult': 4},
            'Stuntman': {'chips': 250},
            'Misprint': {'mult': random.randint(0, 23)},
            'Gros Michel': {'mult': 15},
            'Cavendish': {'x_mult': 3},
            'Half Joker': {'mult': 20} if len(context.get('scoring_cards', [])) <= 3 else None,
            'Abstract Joker': {'mult': 3 * len(game_state.get('jokers', []))},
            'Acrobat': {'x_mult': 3} if game_state.get('hands_left', 1) == 1 else None,
            'Mystic Summit': {'mult': 15} if game_state.get('discards_left', 0) == 0 else None,
            'Banner': {'chips': 30 * game_state.get('discards_left', 0)},
            'Blue Joker': {'chips': 2 * len(game_state.get('deck', []))},
            'Popcorn': {'mult': 20},  # Would decrease each round
            'Ice Cream': {'chips': 100},  # Would decrease each hand
        }
        
        # Suit-specific jokers
        suit_jokers = {
            'Greedy Joker': ('Diamonds', {'mult': 3}),
            'Lusty Joker': ('Hearts', {'mult': 3}),
            'Wrathful Joker': ('Spades', {'mult': 3}),
            'Gluttonous Joker': ('Clubs', {'mult': 3}),
        }
        
        # Hand-type specific jokers
        hand_type_jokers = {
            'Jolly Joker': ('Pair', {'mult': 8}),
            'Zany Joker': ('Three of a Kind', {'mult': 12}),
            'Mad Joker': ('Two Pair', {'mult': 10}),
            'Crazy Joker': ('Straight', {'mult': 12}),
            'Droll Joker': ('Flush', {'mult': 10}),
            'Sly Joker': ('Pair', {'chips': 50}),
            'Wily Joker': ('Three of a Kind', {'chips': 100}),
            'Clever Joker': ('Two Pair', {'chips': 80}),
            'Devious Joker': ('Straight', {'chips': 100}),
            'Crafty Joker': ('Flush', {'chips': 80}),
            'The Duo': ('Pair', {'x_mult': 2}),
            'The Trio': ('Three of a Kind', {'x_mult': 3}),
            'The Family': ('Four of a Kind', {'x_mult': 4}),
            'The Order': ('Straight', {'x_mult': 3}),
            'The Tribe': ('Flush', {'x_mult': 2}),
        }
        
        # Check basic effects
        if joker_name in basic_effects:
            return basic_effects[joker_name]
        
        # Check suit jokers
        if joker_name in suit_jokers:
            suit, effect = suit_jokers[joker_name]
            if self._has_suit(context, suit):
                return effect
        
        # Check hand type jokers
        if joker_name in hand_type_jokers:
            hand_type, effect = hand_type_jokers[joker_name]
            if context.get('hand_type') == hand_type:
                return effect
        
        # Special condition jokers
        if joker_name == 'Blackboard':
            # Check if all cards are spades or clubs
            cards = context.get('cards', [])
            if all(hasattr(card, 'suit') and card.suit in ['Spades', 'Clubs'] for card in cards):
                return {'x_mult': 3}
        
        elif joker_name == 'Seeing Double':
            # Check if hand has clubs and another suit
            suits = set(card.suit for card in context.get('scoring_cards', []) if hasattr(card, 'suit'))
            if 'Clubs' in suits and len(suits) > 1:
                return {'x_mult': 2}
        
        elif joker_name == 'Flower Pot':
            # Check if hand has all four suits
            suits = set(card.suit for card in context.get('scoring_cards', []) if hasattr(card, 'suit'))
            if len(suits) == 4:
                return {'x_mult': 3}
        
        elif joker_name == 'Baron':
            # X1.5 mult per king in hand
            kings = sum(1 for card in context.get('cards', []) if hasattr(card, 'rank') and card.rank == 13)
            if kings > 0:
                return {'x_mult': 1.5 ** kings}
        
        elif joker_name == 'Shoot the Moon':
            # +13 mult per queen in hand
            queens = sum(1 for card in context.get('cards', []) if hasattr(card, 'rank') and card.rank == 12)
            if queens > 0:
                return {'mult': 13 * queens}
        
        return None
    
    def _individual_scoring_effects(self, joker_name: str, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effects that apply to individual cards during scoring"""
        
        card = context.get('card')
        if not card or not hasattr(card, 'rank'):
            return None
        
        # Rank-based effects
        rank_effects = {
            'Fibonacci': ([2, 3, 5, 8, 14], {'mult': 8}),
            'Even Steven': ([2, 4, 6, 8, 10], {'mult': 4}),
            'Odd Todd': ([3, 5, 7, 9, 14], {'chips': 31}),  # Ace counts as odd
            'Scholar': ([14], {'chips': 20, 'mult': 4}),
            'Walkie Talkie': ([4, 10], {'chips': 10, 'mult': 4}),
            'Wee Joker': ([2], {'chips': 8}),
            '8 Ball': ([8], {'special': 'tarot_chance'}),
        }
        
        # Face card effects
        face_effects = {
            'Scary Face': {'chips': 30},
            'Smiley Face': {'mult': 5},
            'Triboulet': {'x_mult': 2} if card.rank in [12, 13] else None,  # Queens and Kings
        }
        
        # Suit-based effects
        suit_effects = {
            'Arrowhead': ('Spades', {'chips': 50}),
            'Onyx Agate': ('Clubs', {'mult': 7}),
            'Rough Gem': ('Diamonds', {'money': 1}),
            'Bloodstone': ('Hearts', {'x_mult': 2} if random.random() < 0.5 else None),
        }
        
        # Check rank effects
        for joker, (ranks, effect) in rank_effects.items():
            if joker_name == joker and card.rank in ranks:
                if effect.get('special') == 'tarot_chance' and random.random() < 0.25:
                    # Would create tarot card
                    return {'message': 'Tarot created!'}
                return effect
        
        # Check face card effects
        if card.rank in [11, 12, 13]:  # Face cards
            for joker, effect in face_effects.items():
                if joker_name == joker and effect:
                    return effect
        
        # Check suit effects
        if hasattr(card, 'suit'):
            for joker, (suit, effect) in suit_effects.items():
                if joker_name == joker and card.suit == suit and effect:
                    return effect
        
        return None
    
    def _discard_effects(self, joker_name: str, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effects that trigger on discard"""
        
        if joker_name == 'Trading Card':
            if context.get('is_first_discard') and len(context.get('discarded_cards', [])) == 1:
                return {'money': 3}
        
        elif joker_name == 'Faceless Joker':
            face_count = sum(1 for card in context.get('discarded_cards', []) 
                           if hasattr(card, 'rank') and card.rank in [11, 12, 13])
            if face_count >= 3:
                return {'money': 5}
        
        elif joker_name == 'Mail-In Rebate':
            # Would check for specific rank
            return None
        
        elif joker_name == 'Green Joker':
            # Lose mult on discard
            if joker_name not in self.joker_states:
                self.joker_states[joker_name] = {'mult': 1}
            self.joker_states[joker_name]['mult'] = max(0, self.joker_states[joker_name]['mult'] - 1)
        
        return None
    
    def _before_scoring_effects(self, joker_name: str, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effects that apply before scoring"""
        
        if joker_name == 'Green Joker':
            # Gain mult on hand played
            if joker_name not in self.joker_states:
                self.joker_states[joker_name] = {'mult': 1}
            self.joker_states[joker_name]['mult'] += 1
            return {'message': f"+1 Mult (now {self.joker_states[joker_name]['mult']})"}
        
        elif joker_name == 'Ride the Bus':
            # Check for face cards
            has_face = any(hasattr(card, 'rank') and card.rank in [11, 12, 13] 
                          for card in context.get('scoring_cards', []))
            if joker_name not in self.joker_states:
                self.joker_states[joker_name] = {'mult': 0}
            
            if has_face:
                self.joker_states[joker_name]['mult'] = 0
                return {'message': 'Reset!'}
            else:
                self.joker_states[joker_name]['mult'] += 1
                return {'message': f"+1 Mult (now {self.joker_states[joker_name]['mult']})"}
        
        return None
    
    def _skip_blind_effects(self, joker_name: str, context: Dict[str, Any], game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effects that trigger when skipping blinds"""
        
        if joker_name == 'Throwback':
            # Would increase X mult
            return {'message': 'X mult increased!'}
        
        return None
    
    def _has_suit(self, context: Dict[str, Any], suit: str) -> bool:
        """Check if any scoring card has the given suit"""
        for card in context.get('scoring_cards', []):
            if hasattr(card, 'suit') and card.suit == suit:
                return True
        return False
    
    def end_of_round_effects(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle end-of-round joker effects"""
        effects = []
        
        # Simplified - just return empty list for now
        # Full implementation would handle Popcorn, Ice Cream, etc.
        
        return effects