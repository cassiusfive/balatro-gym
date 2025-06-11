"""Play phase handler for Balatro RL environment.

This module handles all actions during the PLAY phase including:
- Playing poker hands
- Discarding cards
- Using consumables
- Card selection
"""

from typing import Tuple, Dict, List, Any, Optional
import numpy as np

from balatro_gym.envs.state import UnifiedGameState, CardState
from balatro_gym.envs.rng import DeterministicRNG
from balatro_gym.envs.card_adapter import CardAdapter
from balatro_gym.envs.reward_calculator import RewardCalculator
from balatro_gym.constants import Action, Phase
from balatro_gym.cards import Card, Enhancement, Edition, Seal, EnhancementEffects, SealEffects
from balatro_gym.scoring_engine import ScoreEngine, HandType
from balatro_gym.unified_scoring import UnifiedScorer, ScoringContext
from balatro_gym.complete_joker_effects import CompleteJokerEffects
from balatro_gym.consumables import ConsumableManager
from balatro_gym.boss_blinds import BossBlindManager
from balatro_gym.balatro_game import BalatroGame


class PlayPhaseHandler:
    """Handles all actions during the PLAY phase."""
    
    def __init__(self, 
                 state: UnifiedGameState,
                 game: BalatroGame,
                 engine: ScoreEngine,
                 unified_scorer: UnifiedScorer,
                 joker_effects_engine: CompleteJokerEffects,
                 consumable_manager: ConsumableManager,
                 boss_blind_manager: BossBlindManager,
                 rng: DeterministicRNG):
        """Initialize the play phase handler.
        
        Args:
            state: Game state
            game: Core game instance
            engine: Scoring engine
            unified_scorer: Unified scoring system
            joker_effects_engine: Joker effects processor
            consumable_manager: Consumable effects manager
            boss_blind_manager: Boss blind manager
            rng: RNG system
        """
        self.state = state
        self.game = game
        self.engine = engine
        self.unified_scorer = unified_scorer
        self.joker_effects_engine = joker_effects_engine
        self.consumable_manager = consumable_manager
        self.boss_blind_manager = boss_blind_manager
        self.rng = rng
        self.reward_calculator = RewardCalculator()
    
    def step(self, action: int) -> Tuple[float, bool, Dict]:
        """Process an action during play phase.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (reward, terminated, info)
        """
        if action == Action.PLAY_HAND:
            return self._handle_play_hand()
        elif action == Action.DISCARD:
            return self._handle_discard()
        elif Action.SELECT_CARD_BASE <= action < Action.SELECT_CARD_BASE + Action.SELECT_CARD_COUNT:
            return self._handle_card_selection(action)
        elif Action.USE_CONSUMABLE_BASE <= action < Action.USE_CONSUMABLE_BASE + Action.USE_CONSUMABLE_COUNT:
            return self._handle_consumable_use(action)
        else:
            return -1.0, False, {'error': 'Invalid play phase action'}
    
    def _handle_play_hand(self) -> Tuple[float, bool, Dict]:
        """Handle playing the selected hand."""
        if len(self.state.selected_cards) == 0:
            return -1.0, False, {'error': 'No cards selected'}
        
        # Get selected cards
        selected_cards, selected_game_cards = self._get_selected_cards()
        if not selected_cards:
            return -1.0, False, {'error': 'Invalid card selection'}
        
        # Sync and highlight cards in game
        self._sync_and_highlight_cards()
        
        # Classify the hand
        hand_type, _ = self.game._classify_hand(
            [self.game.deck[i] for i in self.game.highlighted_indexes]
        )
        hand_type_name = hand_type.name.replace('_', ' ').title()
        
        # Check boss blind restrictions
        if not self._check_boss_blind_can_play(selected_game_cards, hand_type_name):
            return -1.0, False, {'error': 'Boss blind prevents playing this hand'}
        
        # Score the hand
        base_score, breakdown = self._score_hand(selected_cards, hand_type, hand_type_name)
        
        # Apply card effects and calculate final score
        final_score, extra_money, cards_to_destroy, consumables_created = \
            self._apply_card_effects(selected_cards, selected_game_cards, base_score)
        
        # Apply boss blind scoring modifications
        final_score = self._apply_boss_blind_scoring(final_score, selected_game_cards, hand_type, hand_type_name)
        
        # Update game state
        self._update_state_after_play(final_score, extra_money, cards_to_destroy, consumables_created)
        
        # Track hand usage
        self.engine.hand_play_counts[hand_type] += 1
        
        # Apply boss blind post-scoring effects
        if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
            self.boss_blind_manager.on_hand_scored(selected_game_cards, hand_type_name, self.state.to_dict())
            # Sync any changes from boss blind
            if 'money' in self.state.to_dict():
                self.state.money = self.state.to_dict()['money']
        
        # Clear selection
        self.state.selected_cards = []
        
        # Calculate reward
        reward_info = self.reward_calculator.calculate_play_reward(
            old_score=final_score - self.state.round_chips_scored,
            new_score=self.state.round_chips_scored,
            chips_needed=self.state.chips_needed,
            final_score=final_score,
            hand_type=hand_type,
            cards_played=len(selected_game_cards),
            ante=self.state.ante,
            hands_left=self.state.hands_left,
            joker_names=[j.name for j in self.state.jokers],
            selected_game_cards=selected_game_cards
        )
        
        reward = reward_info['total_reward']
        info = {
            'score_breakdown': breakdown,
            'final_score': final_score,
            'hand_type': hand_type,
            'cards_played': len(selected_game_cards),
            'reward_breakdown': reward_info
        }
        
        # Check end conditions
        terminated = False
        if self.state.round_chips_scored >= self.state.chips_needed:
            # Beat the blind!
            from balatro_gym.envs.utils.round_manager import RoundManager
            round_manager = RoundManager(self.state, self.game, self.joker_effects_engine)
            round_manager.advance_round()
            info['beat_blind'] = True
        elif self.state.hands_left <= 1:
            # Failed the blind
            terminated = True
            info['failed'] = True
        else:
            # Continue playing
            self.state.hands_left -= 1
            self.game.round_hands = self.state.hands_left
            self._draw_new_hand()
        
        return reward, terminated, info
    
    def _handle_discard(self) -> Tuple[float, bool, Dict]:
        """Handle discarding selected cards."""
        if self.state.discards_left <= 0:
            return -1.0, False, {'error': 'No discards left'}
        
        if len(self.state.selected_cards) == 0:
            return -1.0, False, {'error': 'No cards selected'}
        
        # Get discarded cards
        discarded_cards = []
        purple_seal_count = 0
        
        for idx in self.state.selected_cards:
            if idx < len(self.state.hand_indexes):
                card_idx = self.state.hand_indexes[idx]
                if card_idx < len(self.state.deck):
                    card = self.state.deck[card_idx]
                    card_state = self.state.get_card_state(card_idx)
                    
                    # Track purple seals
                    if card_state.seal == Seal.PURPLE:
                        purple_seal_count += 1
                    
                    # Track card for discard effects
                    card_state.times_discarded += 1
                    
                    # Create card format for joker effects
                    discard_card = type('Card', (), {
                        'rank': card.rank.value,
                        'suit': card.suit.name.title()
                    })
                    discarded_cards.append(discard_card)
        
        # Apply discard joker effects
        money_from_discards = self._apply_discard_effects(discarded_cards)
        
        # Execute discard in game
        self._sync_state_to_game()
        for idx in sorted(self.state.selected_cards, reverse=True):
            if idx < len(self.game.hand_indexes):
                self.game.highlight_card(idx)
        
        self.game.discard_hand()
        self.state.discards_left -= 1
        self.state.cards_discarded_total += len(self.state.selected_cards)
        self.state.selected_cards = []
        self._sync_state_from_game()
        
        # Create tarot cards from purple seals
        tarots_created = self._create_tarots_from_purple_seals(purple_seal_count)
        
        # Calculate reward
        reward = 0.2  # Base discard value
        reward += self._calculate_discard_reward(money_from_discards, len(discarded_cards))
        
        info = {
            'cards_discarded': len(discarded_cards),
            'money_earned': money_from_discards,
            'discards_left': self.state.discards_left
        }
        
        if tarots_created:
            info['tarots_created'] = tarots_created
        
        return reward, False, info
    
    def _handle_card_selection(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle selecting/deselecting a card."""
        card_idx = action - Action.SELECT_CARD_BASE
        
        if card_idx >= len(self.state.hand_indexes):
            return -1.0, False, {'error': 'Invalid card index'}
        
        # Check if card is face down (boss blind effect)
        if card_idx in self.state.face_down_cards:
            return -1.0, False, {'error': 'Cannot select face down card'}
        
        # Toggle selection
        if card_idx in self.state.selected_cards:
            self.state.selected_cards.remove(card_idx)
        else:
            self.state.selected_cards.append(card_idx)
        
        return 0.0, False, {'selected_cards': self.state.selected_cards.copy()}
    
    def _handle_consumable_use(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle using a consumable."""
        consumable_idx = action - Action.USE_CONSUMABLE_BASE
        
        if consumable_idx >= len(self.state.consumables):
            return -1.0, False, {'error': 'Invalid consumable index'}
        
        consumable_name = self.state.consumables[consumable_idx]
        
        # Get target cards
        target_cards = []
        for idx in self.state.selected_cards:
            if idx < len(self.state.hand_indexes):
                card_idx = self.state.hand_indexes[idx]
                if card_idx < len(self.state.deck):
                    card = self.state.deck[card_idx]
                    consumable_card = CardAdapter.to_consumable_format(card, card_idx, self.state)
                    target_cards.append(consumable_card)
        
        # Apply consumable effect
        result = self.consumable_manager.use_consumable(
            consumable_name, self.state.to_dict(), target_cards
        )
        
        if not result['success']:
            return -1.0, False, {'error': result.get('message', 'Failed to use consumable')}
        
        # Remove consumable
        self.state.consumables.pop(consumable_idx)
        
        # Process results
        reward = self._process_consumable_results(result)
        
        # Clear selection
        self.state.selected_cards = []
        
        info = {
            'consumable_used': consumable_name,
            'result': result['message']
        }
        
        return reward, False, info
    
    def apply_boss_blind_to_hand(self):
        """Apply boss blind effects when drawing a new hand."""
        if not self.state.boss_blind_active or not self.boss_blind_manager.active_blind:
            return
        
        hand_cards = [self.state.deck[i] for i in self.state.hand_indexes]
        effects = self.boss_blind_manager.on_hand_drawn(hand_cards, self.state.to_dict())
        
        # Apply face down effects
        if 'face_down_cards' in effects:
            self.state.face_down_cards = effects['face_down_cards']
        
        # Apply forced discards (The Hook)
        if 'discarded_cards' in effects:
            for idx in sorted(effects['discarded_cards'], reverse=True):
                if idx < len(self.state.hand_indexes):
                    self.state.hand_indexes.pop(idx)
            self._sync_state_to_game()
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _get_selected_cards(self) -> Tuple[List[Any], List[Card]]:
        """Get selected cards in both scoring and game formats."""
        selected_cards = []
        selected_game_cards = []
        
        for idx in self.state.selected_cards:
            if idx < len(self.state.hand_indexes):
                card_idx = self.state.hand_indexes[idx]
                if card_idx < len(self.state.deck):
                    card = self.state.deck[card_idx]
                    selected_game_cards.append(card)
                    scoring_card = CardAdapter.to_scoring_format(card, card_idx, self.state)
                    selected_cards.append(scoring_card)
        
        return selected_cards, selected_game_cards
    
    def _sync_and_highlight_cards(self):
        """Sync state to game and highlight selected cards."""
        self._sync_state_to_game()
        for idx in self.state.selected_cards:
            if idx < len(self.game.hand_indexes):
                self.game.highlight_card(idx)
    
    def _check_boss_blind_can_play(self, cards: List[Card], hand_type: str) -> bool:
        """Check if boss blind allows playing this hand."""
        if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
            can_play, _ = self.boss_blind_manager.can_play_hand(cards, hand_type)
            return can_play
        return True
    
    def _score_hand(self, selected_cards: List[Any], hand_type: HandType, 
                    hand_type_name: str) -> Tuple[int, Dict]:
        """Score the selected hand."""
        scoring_context = ScoringContext(
            cards=selected_cards,
            scoring_cards=selected_cards,
            hand_type=hand_type,
            hand_type_name=hand_type_name,
            game_state=self.state.to_dict()
        )
        
        return self.unified_scorer.score_hand(scoring_context)
    
    def _apply_card_effects(self, selected_cards: List[Any], selected_game_cards: List[Card], 
                           base_score: int) -> Tuple[int, int, List[int], List[str]]:
        """Apply card enhancement/edition/seal effects."""
        final_score = base_score
        extra_money = 0
        cards_to_destroy = []
        consumables_to_create = []
        cards_to_retrigger = []
        
        for i, (idx, card, scoring_card) in enumerate(
            zip(self.state.selected_cards, selected_game_cards, selected_cards)
        ):
            if idx < len(self.state.hand_indexes):
                card_idx = self.state.hand_indexes[idx]
                card_state = self.state.get_card_state(card_idx)
                
                # Track card usage
                card_state.times_played += 1
                card_state.times_scored += 1
                
                # Apply enhancement effects
                if card_state.enhancement == Enhancement.GLASS:
                    if self.rng.get_float('card_enhancement') < 0.25:
                        cards_to_destroy.append(card_idx)
                elif card_state.enhancement == Enhancement.LUCKY:
                    mult_roll = self.rng.get_float('card_enhancement')
                    money_roll = self.rng.get_float('card_enhancement')
                    lucky_mult, lucky_money = EnhancementEffects.get_lucky_bonus(mult_roll, money_roll)
                    if lucky_money > 0:
                        extra_money += lucky_money
                
                # Apply seal effects
                if card_state.seal == Seal.GOLD:
                    extra_money += SealEffects.get_money_bonus(card_state.seal)
                elif card_state.seal == Seal.RED:
                    cards_to_retrigger.append(i)
                elif card_state.seal == Seal.BLUE:
                    # Create planet based on hand type
                    planet_map = {
                        'High Card': 'Pluto', 'One Pair': 'Mercury', 'Two Pair': 'Venus',
                        'Three Kind': 'Earth', 'Straight': 'Mars', 'Flush': 'Jupiter',
                        'Full House': 'Saturn', 'Four Kind': 'Uranus', 
                        'Straight Flush': 'Neptune', 'Five Kind': 'Planet X'
                    }
                    hand_name = scoring_card.hand_type.name.replace('_', ' ').title()
                    planet = planet_map.get(hand_name)
                    if planet and len(self.state.consumables) < self.state.consumable_slots:
                        consumables_to_create.append(planet)
        
        # Apply steel card bonus
        steel_mult = self._calculate_steel_bonus()
        final_score = int(final_score * steel_mult)
        
        # Apply retriggers
        retrigger_bonus = len(cards_to_retrigger) * 0.5
        final_score = int(final_score * (1 + retrigger_bonus))
        
        return final_score, extra_money, cards_to_destroy, consumables_to_create
    
    def _apply_boss_blind_scoring(self, score: int, cards: List[Card], 
                                  hand_type: HandType, hand_type_name: str) -> int:
        """Apply boss blind scoring modifications."""
        if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
            base_chips, base_mult = self.engine.get_hand_chips_mult(hand_type)
            modified_chips, modified_mult = self.boss_blind_manager.modify_scoring(
                base_chips, base_mult, cards, hand_type_name
            )
            
            if base_chips > 0 and base_mult > 0:
                chip_ratio = modified_chips / base_chips
                mult_ratio = modified_mult / base_mult
                score = int(score * chip_ratio * mult_ratio)
        
        return score
    
    def _update_state_after_play(self, score: int, extra_money: int, 
                                cards_to_destroy: List[int], consumables: List[str]):
        """Update game state after playing a hand."""
        # Update scores
        self.state.round_chips_scored += score
        self.state.chips_scored += score
        self.state.hands_played_total += 1
        self.state.hands_played_ante += 1
        self.state.best_hand_this_ante = max(self.state.best_hand_this_ante, score)
        
        # Add money
        self.state.money += extra_money
        
        # Create consumables
        for consumable in consumables:
            if len(self.state.consumables) < self.state.consumable_slots:
                self.state.consumables.append(consumable)
        
        # Destroy cards
        for card_idx in cards_to_destroy:
            card_state = self.state.get_card_state(card_idx)
            card_state.is_destroyed = True
    
    def _draw_new_hand(self):
        """Draw a new hand of cards."""
        self.game._draw_cards()
        self.state.hand_indexes = self.game.hand_indexes.copy()
        
        # Apply boss blind effects to new hand
        if self.state.boss_blind_active:
            self.apply_boss_blind_to_hand()
        
        # Handle forced draw count (The Serpent)
        if self.state.force_draw_count is not None:
            self._apply_forced_draw_count()
    
    def _apply_forced_draw_count(self):
        """Apply forced draw count from boss blind."""
        while len(self.state.hand_indexes) > self.state.force_draw_count:
            self.state.hand_indexes.pop()
        
        while len(self.state.hand_indexes) < self.state.force_draw_count:
            available = [i for i in range(len(self.state.deck)) 
                        if i not in self.state.hand_indexes]
            if available:
                self.state.hand_indexes.append(self.rng.choice('card_draw', available))
        
        self.state.force_draw_count = None
    
    def _apply_discard_effects(self, discarded_cards: List[Any]) -> int:
        """Apply joker effects for discarding."""
        discard_context = {
            'phase': 'discard',
            'discarded_cards': discarded_cards,
            'last_discarded_card': discarded_cards[-1] if discarded_cards else None,
            'is_first_discard': self.state.discards_left == self.game.discards
        }
        
        money_earned = 0
        for joker in self.state.jokers:
            effect = self.joker_effects_engine.apply_joker_effect(
                type('Joker', (), {'name': joker.name}), 
                discard_context, 
                self.state.to_dict()
            )
            if effect and 'money' in effect:
                money_earned += effect['money']
                self.state.money += effect['money']
        
        return money_earned
    
    def _create_tarots_from_purple_seals(self, count: int) -> List[str]:
        """Create tarot cards from purple seals."""
        if count == 0:
            return []
        
        tarots = [
            'The Fool', 'The Magician', 'The High Priestess', 'The Empress',
            'The Emperor', 'The Hierophant', 'The Lovers', 'The Chariot',
            'Strength', 'The Hermit', 'Wheel of Fortune', 'Justice',
            'The Hanged Man', 'Death', 'Temperance', 'The Devil',
            'The Tower', 'The Star', 'The Moon', 'The Sun',
            'Judgement', 'The World'
        ]
        
        created = []
        for _ in range(count):
            if len(self.state.consumables) < self.state.consumable_slots:
                tarot = self.rng.choice('seal_applications', tarots)
                self.state.consumables.append(tarot)
                created.append(tarot)
        
        return created
    
    def _calculate_discard_reward(self, money_earned: int, cards_discarded: int) -> float:
        """Calculate reward for discarding."""
        reward = 0.0
        
        # Reward for discard synergies
        discard_jokers = ['Faceless Joker', 'Hit the Road', 'Reserved Parking', 'Luchador']
        discard_joker_count = sum(1 for j in self.state.jokers if j.name in discard_jokers)
        if discard_joker_count > 0:
            reward += 0.5 * discard_joker_count
        
        # Money earned bonus
        if money_earned > 0:
            reward += money_earned / 5.0
        
        # Strategic discard bonus
        progress = self.state.round_chips_scored / max(1, self.state.chips_needed)
        if progress < 0.5 and self.state.discards_left > 1:
            reward += 0.5  # Encourage discarding when behind
        elif progress > 0.8 and self.state.discards_left > 1:
            reward -= 0.3  # Discourage wasteful discards when ahead
        
        return reward
    
    def _process_consumable_results(self, result: Dict) -> float:
        """Process consumable use results and calculate reward."""
        reward = 0.0
        
        # Money gained
        if result.get('money_gained', 0) > 0:
            self.state.money += result['money_gained']
            reward += result['money_gained'] / 10.0
        
        # Planet used
        if result.get('planet_used'):
            reward += self._apply_planet_effect(result['planet_used'])
        
        # Cards affected
        if result.get('cards_affected'):
            reward += self._apply_card_modifications(result['cards_affected'])
        
        # Other effects
        if result.get('cards_created'):
            reward += len(result['cards_created']) * 3.0
        
        if result.get('cards_destroyed'):
            reward += len(result['cards_destroyed']) * 1.0
        
        if result.get('jokers_created'):
            reward += self._create_jokers(result['jokers_created'])
        
        if result.get('items_created'):
            reward += self._create_items(result['items_created'])
        
        if result.get('hand_size_change'):
            self.state.hand_size += result['hand_size_change']
            self.game.hand_size = self.state.hand_size
            reward += abs(result['hand_size_change']) * 5.0
        
        return reward
    
    def _apply_planet_effect(self, planet: str) -> float:
        """Apply planet effect to hand levels."""
        planet_map = {
            'Mercury': HandType.ONE_PAIR,
            'Venus': HandType.TWO_PAIR,
            'Earth': HandType.THREE_KIND,
            'Mars': HandType.STRAIGHT,
            'Jupiter': HandType.FLUSH,
            'Saturn': HandType.FULL_HOUSE,
            'Uranus': HandType.FOUR_KIND,
            'Neptune': HandType.STRAIGHT_FLUSH,
            'Pluto': HandType.HIGH_CARD,
            'Planet X': HandType.FIVE_KIND,
        }
        
        if planet in planet_map:
            hand_type = planet_map[planet]
            self.engine.apply_planet(hand_type)
            self.state.hand_levels[hand_type] = self.state.hand_levels.get(hand_type, 0) + 1
            return 10.0
        
        return 0.0
    
    def _apply_card_modifications(self, affected_cards: List[Any]) -> float:
        """Apply modifications to affected cards."""
        for affected in affected_cards:
            if hasattr(affected, 'card_idx'):
                card_state = self.state.get_card_state(affected.card_idx)
                
                if hasattr(affected, 'enhancement'):
                    card_state.enhancement = affected.enhancement
                if hasattr(affected, 'edition'):
                    card_state.edition = affected.edition
                if hasattr(affected, 'seal'):
                    card_state.seal = affected.seal
        
        return len(affected_cards) * 2.0
    
    def _create_jokers(self, joker_names: List[str]) -> float:
        """Create jokers from consumable effects."""
        created = 0
        for joker_name in joker_names:
            if len(self.state.jokers) < self.state.joker_slots:
                from balatro_gym.jokers import JOKER_LIBRARY
                for joker_info in JOKER_LIBRARY:
                    if joker_info.name == joker_name:
                        self.state.jokers.append(joker_info)
                        created += 1
                        break
        
        return created * 15.0
    
    def _create_items(self, items: List[str]) -> float:
        """Create consumable items."""
        created = 0
        for item in items:
            if len(self.state.consumables) < self.state.consumable_slots:
                self.state.consumables.append(item)
                created += 1
        
        return created * 5.0
    
    def _calculate_steel_bonus(self) -> float:
        """Calculate mult multiplier from steel cards remaining in hand."""
        steel_mult = 1.0
        selected_hand_indexes = {self.state.hand_indexes[i] for i in self.state.selected_cards 
                                if i < len(self.state.hand_indexes)}
        
        for idx in self.state.hand_indexes:
            if idx not in selected_hand_indexes:
                card_state = self.state.get_card_state(idx)
                if card_state.enhancement == Enhancement.STEEL:
                    steel_mult *= EnhancementEffects.get_mult_multiplier(
                        Enhancement.STEEL, in_hand=True
                    )
        
        return steel_mult
    
    def _sync_state_to_game(self):
        """Sync state to game instance."""
        self.game.deck = self.state.deck
        self.game.hand_indexes = self.state.hand_indexes
        self.game.round_hands = self.state.hands_left
        self.game.round_discards = self.state.discards_left
        self.game.round_score = self.state.chips_scored
    
    def _sync_state_from_game(self):
        """Sync state from game instance."""
        current_total_score = self.state.chips_scored
        current_round_score = self.state.round_chips_scored
        
        self.state.deck = self.game.deck
        self.state.hand_indexes = self.game.hand_indexes
        self.state.hands_left = self.game.round_hands
        self.state.discards_left = self.game.round_discards
        
        self.state.chips_scored = current_total_score
        self.state.round_chips_scored = current_round_score
