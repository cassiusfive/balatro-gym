"""balatro_gym/envs/balatro_env_v2.py – Integrated with complete simulator

Uses the complete BalatroSimulator for hand evaluation and scoring, with all
150+ joker implementations and proper card enhancement/edition effects.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from balatro_gym.shop import Shop, ShopAction, PlayerState, ItemType
from balatro_gym.scoring_engine import ScoreEngine, Planet
from balatro_gym.balatro_game import BalatroGame
from balatro_gym.balatro_sim import BalatroSimulator
from balatro_gym.complete_joker_effects import CompleteJokerEffects

# ---------------------------------------------------------------------------
# Card class for compatibility
# ---------------------------------------------------------------------------
@dataclass
class Card:
    rank: int  # 2-14 (14 = Ace)
    suit: str  # 'Spades', 'Hearts', 'Diamonds', 'Clubs'
    enhancement: Optional[str] = None
    edition: Optional[str] = None
    seal: Optional[str] = None
    base_value: Optional[int] = None
    
    def __post_init__(self):
        if self.base_value is None:
            if self.rank == 14:  # Ace
                self.base_value = 11
            elif self.rank >= 11:  # Face cards
                self.base_value = 10
            else:
                self.base_value = self.rank

# ---------------------------------------------------------------------------
# Joker Modifier Adapter
# ---------------------------------------------------------------------------
class JokerModifierAdapter:
    """Adapter to use CompleteJokerEffects with ScoreEngine"""
    def __init__(self, joker_name: str, simulator: BalatroSimulator):
        self.joker_name = joker_name
        self.simulator = simulator
        self.joker_obj = type('obj', (object,), {'name': joker_name})()
        
    def __call__(self, score: float, hand: List[int], eng: ScoreEngine) -> float:
        # This adapter is for ScoreEngine compatibility
        # We'll use the simulator directly instead
        return score

# ---------------------------------------------------------------------------
# Env constants
# ---------------------------------------------------------------------------
PLAY_MAX_ID   = 9
SHOP_MIN_ID   = 10
ACTION_SPACE_SIZE = 70
PHASE_PLAY = 0
PHASE_SHOP = 1

# ---------------------------------------------------------------------------
class BalatroEnvComplete(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # Initialize complete simulator
        self.simulator = BalatroSimulator()

        # spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            hand=spaces.Box(-1, 51, (8,), dtype=np.int8),
            chips=spaces.Box(0, 1_000_000_000, (), dtype=np.int32),
            phase=spaces.Discrete(2),
            action_mask=spaces.MultiBinary(ACTION_SPACE_SIZE),
            ante=spaces.Box(1, 8, (), dtype=np.int8),
            hands_left=spaces.Box(0, 4, (), dtype=np.int8),
            discards_left=spaces.Box(0, 3, (), dtype=np.int8),
            joker_slots=spaces.Box(-1, 150, (5,), dtype=np.int16),
            # Hand possibility indicators
            has_pair=spaces.Discrete(2),
            has_two_pair=spaces.Discrete(2),
            has_three_kind=spaces.Discrete(2),
            has_straight=spaces.Discrete(2),
            has_flush=spaces.Discrete(2),
        )

        # core objects
        self.engine = ScoreEngine()  # Keep for planet cards
        self.game   = BalatroGame(engine=self.engine)
        self.player = PlayerState(chips=100)
        self.ante   = 1
        self.hands_left = 4
        self.discards_left = 3
        
        # Sync simulator state
        self.simulator.player_state = self.player
        self.simulator.current_ante = self.ante

        self.phase = PHASE_PLAY
        self.shop: Optional[Shop] = None
        self._deal()

    # ------------------------------------------------------------------
    # Reset & helpers
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        # Reset simulator
        self.simulator = BalatroSimulator()
        
        # Reset game components
        self.engine = ScoreEngine()
        self.game   = BalatroGame(engine=self.engine)
        self.player = PlayerState(chips=100)
        self.ante = 1
        self.hands_left = 4
        self.discards_left = 3
        
        # Sync simulator
        self.simulator.player_state = self.player
        self.simulator.current_ante = self.ante
        
        self.phase = PHASE_PLAY
        self.shop  = None
        self._deal()
        return self._obs(), {}

    def _deal(self):
        self.game._draw_cards()
        self.hand = np.array(self.game.hand_indexes[:8], dtype=np.int8)

    def _indexes_to_cards(self, indexes: List[int]) -> List[Card]:
        """Convert card indexes to Card objects"""
        cards = []
        for idx in indexes:
            if idx >= 0:
                rank = (idx % 13) + 2
                suit = ['Spades', 'Hearts', 'Diamonds', 'Clubs'][idx // 13]
                cards.append(Card(rank=rank, suit=suit))
        return cards
    
    def _hand_to_cards(self) -> List[Card]:
        """Convert current hand to Card objects"""
        return self._indexes_to_cards([idx for idx in self.hand if idx >= 0])

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.phase == PHASE_SHOP:
            return self._step_shop(action)
        return self._step_play(action)

    # ---------------- SHOP phase ----------------
    def _step_shop(self, action: int):
        verb, idx = ShopAction.decode(action)
        
        # Track what was purchased before the step
        purchased_item = None
        if verb.startswith("buy_") and idx < len(self.shop.inventory):
            purchased_item = self.shop.inventory[idx]
        
        # The shop.step only takes action_id, not player
        reward, done_shop, info = self.shop.step(action)

        # Handle different purchase types
        if purchased_item:
            if purchased_item.item_type == ItemType.JOKER:
                # Add joker to simulator
                joker_id = purchased_item.payload.get("joker_id")
                if joker_id and joker_id not in self.simulator.player_state.jokers:
                    self.simulator.player_state.jokers.append(joker_id)
                    
            elif purchased_item.item_type == ItemType.PACK:
                # Handle pack opening - info might contain new cards
                if "new_cards" in info:
                    # Cards were added to deck
                    pass

        # Handle planet cards from info
        if "planet" in info:
            self.engine.apply_consumable(info["planet"])
            # Also apply to simulator
            # self.simulator.apply_planet_card(planet_name)

        if done_shop:
            self.phase = PHASE_PLAY
            self.hands_left = 4
            self.discards_left = 3
            self._deal()
            
        return self._obs(), reward, False, False, info
    # ---------------- PLAY phase ----------------
    def _step_play(self, action: int):
        reward = 0.0
        info = {}
        
        if action <= 5:  # Play actions (0-5 for different strategies)
            # Convert hand to cards
            cards = self._hand_to_cards()
            
            # Use simulator to evaluate hand
            hand_result = self.simulator.evaluate_hand(cards)
            
            # Select cards based on action
            if action == 0:  # Play best detected hand
                hand_type = hand_result['top']
                scoring_cards = hand_result[hand_type][0] if hand_result[hand_type] else cards[:5]
            elif action == 1:  # Play first 5
                scoring_cards = cards[:5]
            elif action == 2:  # Play pair if exists
                scoring_cards = hand_result['Pair'][0] if hand_result['Pair'] else cards[:2]
            elif action == 3:  # Play two pair if exists
                scoring_cards = hand_result['Two Pair'][0] if hand_result['Two Pair'] else cards[:4]
            elif action == 4:  # Play three of a kind if exists
                scoring_cards = hand_result['Three of a Kind'][0] if hand_result['Three of a Kind'] else cards[:3]
            elif action == 5:  # Play flush if exists
                scoring_cards = hand_result['Flush'][0] if hand_result['Flush'] else cards[:5]
            
            # Calculate score using simulator
            score, updated_state = self.simulator.calculate_score(scoring_cards)
            reward = score / 100.0  # Scale for RL
            
            # Update game state
            self.player.chips += score
            if 'money' in updated_state:
                self.player.chips = updated_state['money']
            
            # Update hand tracking
            self.hands_left -= 1
            info['hand_type'] = hand_result['top']
            info['score'] = score
            
            if self.hands_left <= 0:
                self._end_round()
            else:
                self._deal()
                
        elif action in [6, 7, 8]:  # Discard actions
            if self.discards_left > 0:
                # Determine which cards to discard
                cards = self._hand_to_cards()
                if action == 6:  # Discard first 1
                    discarded_cards = cards[:1]
                elif action == 7:  # Discard first 3
                    discarded_cards = cards[:3]
                elif action == 8:  # Discard all
                    discarded_cards = cards
                
                # Apply discard effects
                game_state = {'money': self.player.chips}
                self.simulator.apply_joker_discard_effects(discarded_cards, game_state)
                self.player.chips = game_state['money']
                
                self.discards_left -= 1
                reward = -0.1  # Small penalty for discarding
                
                self.game.discard_hand()
                self._deal()
            else:
                reward = -1  # Penalty for invalid discard
                
        return self._obs(), reward, False, False, info

    # -------------- round → shop --------------
    def _end_round(self):
        # Apply end of round joker effects
        game_state = self.simulator._create_game_state()
        effects = self.simulator.apply_joker_end_round_effects(game_state)
        
        # Update money from effects
        if 'money' in game_state:
            self.player.chips = game_state['money']
        
        # Progress to shop
        self.phase = PHASE_SHOP
        self.ante += 1
        self.simulator.current_ante = self.ante
        
        # Blind reward
        blind_reward = self._calculate_blind_reward()
        self.player.chips += blind_reward
        
        self.shop = Shop(self.ante, self.player, seed=int(np.random.randint(1<<31)))

    def _calculate_blind_reward(self) -> int:
        """Calculate money reward for beating blind"""
        base_rewards = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10}
        return base_rewards.get(self.ante, 10) + self.ante

    # -------------- observation ---------------
    def _mask(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == PHASE_PLAY:
            mask[:6] = 1  # Play actions
            if self.discards_left > 0:
                mask[6:9] = 1  # Discard actions
        else:
            mask[ShopAction.SKIP] = 1
            mask[ShopAction.REROLL] = 1
            # Shop buy actions
            for i in range(len(self.shop.inventory)):
                if self.shop.inventory[i].cost <= self.player.chips:
                    mask[ShopAction.BUY_PACK_BASE + i] = 1
        return mask

    def _obs(self):
        obs = {
            "hand": self.hand.copy(),
            "chips": np.int32(self.player.chips),
            "phase": np.int8(self.phase),
            "action_mask": self._mask(),
            "ante": np.int8(self.ante),
            "hands_left": np.int8(self.hands_left),
            "discards_left": np.int8(self.discards_left),
        }
        
        # Add joker information
        joker_ids = self.simulator.player_state.jokers[:5]  # Max 5 jokers
        joker_array = np.full(5, -1, dtype=np.int8)
        joker_array[:len(joker_ids)] = joker_ids
        obs["joker_slots"] = joker_array
        
        # Add hand evaluation hints
        obs["has_pair"] = 0
        obs["has_two_pair"] = 0
        obs["has_three_kind"] = 0
        obs["has_straight"] = 0
        obs["has_flush"] = 0
        
        if self.phase == PHASE_PLAY:
            cards = self._hand_to_cards()
            if cards:
                hand_result = self.simulator.evaluate_hand(cards)
                obs["has_pair"] = int(bool(hand_result.get('Pair')))
                obs["has_two_pair"] = int(bool(hand_result.get('Two Pair')))
                obs["has_three_kind"] = int(bool(hand_result.get('Three of a Kind')))
                obs["has_straight"] = int(bool(hand_result.get('Straight')))
                obs["has_flush"] = int(bool(hand_result.get('Flush')))
        
        if self.phase == PHASE_SHOP and self.shop:
            obs.update(self.shop.get_observation())
            
        return obs

    # -------------- render --------------------
    def render(self):
        if self.render_mode != "human":
            return
            
        print(f"\n=== Ante {self.ante} - {'SHOP' if self.phase == PHASE_SHOP else 'PLAY'} Phase ===")
        print(f"Chips: ${self.player.chips}")
        
        if self.phase == PHASE_SHOP and self.shop:
            print("\nShop Items:")
            for i, item in enumerate(self.shop.inventory):
                affordable = "✓" if item.cost <= self.player.chips else "✗"
                print(f"  [{i}] {item.name:<20} ${item.cost:>4} {affordable}")
            print(f"\nReroll cost: ${int(self.shop.reroll_cost * self.shop._cost_mult())}")
            
        else:
            # Show hand
            suit_symbols = "♠♥♦♣"
            rank_symbols = "23456789TJQKA"
            print("\nHand:")
            for i, idx in enumerate(self.hand):
                if idx >= 0:
                    rank = rank_symbols[idx % 13]
                    suit = suit_symbols[idx // 13]
                    print(f"  [{i}] {rank}{suit}", end="  ")
            print()
            
            # Show possible hands
            cards = self._hand_to_cards()
            if cards:
                hand_result = self.simulator.evaluate_hand(cards)
                print(f"\nBest hand: {hand_result['top']}")
                print(f"Hands left: {self.hands_left}, Discards left: {self.discards_left}")
            
            # Show jokers
            if self.simulator.player_state.jokers:
                print("\nJokers:")
                for jid in self.simulator.player_state.jokers:
                    joker_info = self.simulator.joker_id_to_info.get(jid)
                    if joker_info:
                        print(f"  - {joker_info.name}")

    def close(self):
        pass
