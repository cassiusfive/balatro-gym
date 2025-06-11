# balatro_sim.py - Refactored version
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from scoring_engine import ScoreEngine, HandType
from balatro_gym.planets import Planet, PLANET_MULT
from balatro_gym.jokers import JOKER_LIBRARY, JokerInfo
from balatro_gym.shop import Shop, PlayerState, ShopAction
from balatro_gym.complete_joker_effects import CompleteJokerEffects


@dataclass
class Card:
    rank: int  # 2-14 (14 = Ace)
    suit: str  # 'Spades', 'Hearts', 'Diamonds', 'Clubs'
    base_value: int = 0
    enhancement: Optional[str] = None
    edition: Optional[str] = None
    seal: Optional[str] = None


class BalatroSimulator:
    def __init__(self):
        # Core systems
        self.score_engine = ScoreEngine()
        self.joker_effects = CompleteJokerEffects()
        
        # Player state (managed by shop)
        self.player_state = PlayerState(chips=100)
        
        # Game state
        self.current_ante = 1
        self.hands_played = 0
        self.discards_remaining = 3
        
        # Tarot cards effects
        self.tarots = {
            'The Fool': 'create_tarot',
            'The Magician': 'enhance_cards',
            'The High Priestess': 'create_planet', 
            'The Empress': 'enhance_cards',
            'The Emperor': 'create_tarot',
            'The Hierophant': 'enhance_cards',
            'The Lovers': 'enhance_cards',
            'The Chariot': 'enhance_cards', 
            'Strength': 'modify_cards',
            'The Hermit': 'money',
            'Wheel of Fortune': 'modify_cards',
            'Justice': 'enhance_cards',
            'The Hanged Man': 'destroy_cards',
            'Death': 'destroy_cards',
            'Temperance': 'money',
            'The Devil': 'enhance_cards',
            'The Tower': 'enhance_cards',
            'The Star': 'modify_cards',
            'The Moon': 'modify_cards',
            'The Sun': 'modify_cards',
            'Judgement': 'create_planet',
            'The World': 'modify_cards'
        }
        
        # Card enhancements
        self.enhancements = {
            'bonus': {'chips': 30},
            'mult': {'mult': 4},
            'wild': {'acts_as_any_suit': True},
            'glass': {'x_mult': 2, 'destroy_chance': 0.25},
            'steel': {'x_mult': 1.5, 'permanent': True},
            'stone': {'chips': 50, 'no_suit_rank': True},
            'gold': {'money': 3, 'when_played': True},
            'lucky': {'money_chance': 0.2, 'money_amount': 1}
        }
        
        # Card editions
        self.editions = {
            'foil': {'chips': 50},
            'holographic': {'mult': 10}, 
            'polychrome': {'x_mult': 1.5},
            'negative': {'joker_slot': 1}
        }
        
        # Card seals
        self.seals = {
            'red': {'retrigger': 1},
            'blue': {'create_planet_if_held': True},
            'gold': {'money': 3, 'when_played': True},
            'purple': {'create_tarot_when_discarded': True}
        }
        
        # Create joker name/ID mappings
        self._create_joker_mappings()
        
        # Register base joker modifiers
        self._register_base_modifiers()
        
    def _create_joker_mappings(self):
        """Map between joker IDs and names"""
        self.joker_id_to_info = {j.id: j for j in JOKER_LIBRARY}
        self.joker_name_to_id = {j.name: j.id for j in JOKER_LIBRARY}
        
    def _register_base_modifiers(self):
        """Register permanent modifiers like Four Fingers"""
        # This will be called to set up utility jokers
        pass

    # ===== HAND EVALUATION METHODS =====
    def get_x_same(self, num: int, hand: List[Card]) -> List[List[Card]]:
        """Find cards of the same rank"""
        vals = [[] for _ in range(15)]
        
        for i in range(len(hand) - 1, -1, -1):
            curr = [hand[i]]
            for j in range(len(hand)):
                if hand[i].rank == hand[j].rank and i != j:
                    curr.append(hand[j])
            
            if len(curr) == num:
                vals[hand[i].rank] = curr
        
        ret = []
        for i in range(len(vals) - 1, -1, -1):
            if vals[i]:
                ret.append(vals[i])
        
        return ret

    def get_flush(self, hand: List[Card], four_fingers: bool = False) -> List[List[Card]]:
        """Find flush"""
        ret = []
        suits = ["Spades", "Hearts", "Clubs", "Diamonds"]
        required_cards = 4 if four_fingers else 5
        
        if len(hand) > 5 or len(hand) < required_cards:
            return ret
            
        for suit in suits:
            t = []
            flush_count = 0
            for card in hand:
                if card.suit == suit:
                    flush_count += 1
                    t.append(card)
            
            if flush_count >= required_cards:
                ret.append(t)
                return ret
        
        return ret

    def get_straight(self, hand: List[Card], four_fingers: bool = False, shortcut: bool = False) -> List[List[Card]]:
        """Find straight"""
        ret = []
        required_cards = 4 if four_fingers else 5
        
        if len(hand) > 5 or len(hand) < required_cards:
            return ret
            
        t = []
        ids = {}
        
        for card in hand:
            rank = card.rank
            if 1 < rank < 15:
                if rank in ids:
                    ids[rank].append(card)
                else:
                    ids[rank] = [card]
        
        straight_length = 0
        straight = False
        can_skip = shortcut
        skipped_rank = False
        
        for i in range(14, 1, -1):
            if ids.get(i):
                straight_length += 1
                for card in ids[i]:
                    t.append(card)
            else:
                if can_skip and not skipped_rank:
                    skipped_rank = True
                else:
                    straight_length = 0
                    t = []
                    skipped_rank = False
                    
            if straight_length >= required_cards:
                straight = True
                break
        
        # Check wheel straight (A-2-3-4-5)
        if not straight:
            wheel_cards = []
            wheel_length = 0
            
            for rank in [14, 2, 3, 4, 5]:
                if ids.get(rank):
                    wheel_length += 1
                    wheel_cards.extend(ids[rank])
                else:
                    if can_skip and not skipped_rank:
                        skipped_rank = True
                    else:
                        break
                        
            if wheel_length >= required_cards:
                t = wheel_cards
                straight = True
        
        if straight:
            ret.append(t[:required_cards])
            
        return ret

    def get_highest(self, hand: List[Card]) -> List[List[Card]]:
        """Get highest cards for high card hand"""
        return [hand]

    def evaluate_hand(self, cards: List[Card]) -> Dict:
        """Evaluate poker hand"""
        results = {
            "Flush Five": [],
            "Flush House": [],
            "Five of a Kind": [],
            "Straight Flush": [],
            "Four of a Kind": [],
            "Full House": [],
            "Flush": [],
            "Straight": [],
            "Three of a Kind": [],
            "Two Pair": [],
            "Pair": [],
            "High Card": [],
            "top": None
        }

        # Check for utility jokers
        four_fingers = any(self.joker_id_to_info[jid].name == "Four Fingers" 
                          for jid in self.player_state.jokers)
        shortcut = any(self.joker_id_to_info[jid].name == "Shortcut" 
                      for jid in self.player_state.jokers)

        # Get all the parts
        parts = {
            "_5": self.get_x_same(5, cards),
            "_4": self.get_x_same(4, cards),
            "_3": self.get_x_same(3, cards),
            "_2": self.get_x_same(2, cards),
            "_flush": self.get_flush(cards, four_fingers),
            "_straight": self.get_straight(cards, four_fingers, shortcut),
            "_highest": self.get_highest(cards)
        }

        # Evaluate hands in priority order
        
        # Flush Five
        if parts["_5"] and parts["_flush"]:
            results["Flush Five"] = parts["_5"]
            if not results["top"]:
                results["top"] = "Flush Five"

        # Flush House
        if parts["_3"] and parts["_2"] and parts["_flush"]:
            fh_hand = []
            fh_3 = parts["_3"][0]
            fh_2 = parts["_2"][0]
            fh_hand.extend(fh_3)
            fh_hand.extend(fh_2)
            results["Flush House"].append(fh_hand)
            if not results["top"]:
                results["top"] = "Flush House"

        # Five of a Kind
        if parts["_5"]:
            results["Five of a Kind"] = parts["_5"]
            if not results["top"]:
                results["top"] = "Five of a Kind"

        # Straight Flush
        if parts["_flush"] and parts["_straight"]:
            _s, _f = parts["_straight"], parts["_flush"]
            ret = []
            
            for card in _f[0]:
                ret.append(card)
            
            for card in _s[0]:
                if card not in _f[0]:
                    ret.append(card)
            
            results["Straight Flush"] = [ret]
            if not results["top"]:
                results["top"] = "Straight Flush"

        # Four of a Kind
        if parts["_4"]:
            results["Four of a Kind"] = parts["_4"]
            if not results["top"]:
                results["top"] = "Four of a Kind"

        # Full House
        if parts["_3"] and parts["_2"]:
            fh_hand = []
            fh_3 = parts["_3"][0]
            fh_2 = parts["_2"][0]
            fh_hand.extend(fh_3)
            fh_hand.extend(fh_2)
            results["Full House"].append(fh_hand)
            if not results["top"]:
                results["top"] = "Full House"

        # Flush
        if parts["_flush"]:
            results["Flush"] = parts["_flush"]
            if not results["top"]:
                results["top"] = "Flush"

        # Straight
        if parts["_straight"]:
            results["Straight"] = parts["_straight"]
            if not results["top"]:
                results["top"] = "Straight"

        # Three of a Kind
        if parts["_3"]:
            results["Three of a Kind"] = parts["_3"]
            if not results["top"]:
                results["top"] = "Three of a Kind"

        # Two Pair
        if (len(parts["_2"]) == 2) or (len(parts["_3"]) == 1 and len(parts["_2"]) == 1):
            fh_hand = []
            r = parts["_2"]
            fh_2a = r[0]
            fh_2b = r[1] if len(r) > 1 else parts["_3"][0]
            
            fh_hand.extend(fh_2a)
            fh_hand.extend(fh_2b)
            results["Two Pair"].append(fh_hand)
            if not results["top"]:
                results["top"] = "Two Pair"

        # Pair
        if parts["_2"]:
            results["Pair"] = parts["_2"]
            if not results["top"]:
                results["top"] = "Pair"

        # High Card
        if parts["_highest"]:
            results["High Card"] = parts["_highest"]
            if not results["top"]:
                results["top"] = "High Card"

        # Cascade lower hands
        if results["Five of a Kind"]:
            results["Four of a Kind"] = [results["Five of a Kind"][0][:4]]

        if results["Four of a Kind"]:
            results["Three of a Kind"] = [results["Four of a Kind"][0][:3]]

        if results["Three of a Kind"]:
            results["Pair"] = [results["Three of a Kind"][0][:2]]

        return results

    # ===== CONVERSION UTILITIES =====
    def _card_to_id(self, card: Card) -> int:
        """Convert card object to 0-51 ID"""
        suit_map = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}
        return suit_map[card.suit] * 13 + (card.rank - 2)
    
    def _id_to_card(self, card_id: int) -> Card:
        """Convert 0-51 ID to card object"""
        suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        suit = suits[card_id // 13]
        rank = (card_id % 13) + 2
        return Card(rank=rank, suit=suit, base_value=rank if rank <= 10 else 10)
    
    def _hand_type_to_enum(self, hand_type_str: str) -> HandType:
        """Convert hand type string to enum"""
        mapping = {
            'High Card': HandType.HIGH_CARD,
            'Pair': HandType.ONE_PAIR,
            'Two Pair': HandType.TWO_PAIR,
            'Three of a Kind': HandType.THREE_KIND,
            'Straight': HandType.STRAIGHT,
            'Flush': HandType.FLUSH,
            'Full House': HandType.FULL_HOUSE,
            'Four of a Kind': HandType.FOUR_KIND,
            'Straight Flush': HandType.STRAIGHT_FLUSH,
            'Five of a Kind': HandType.FOUR_KIND,  # Treat as Four Kind
            'Flush Five': HandType.STRAIGHT_FLUSH,  # Treat as SF
            'Flush House': HandType.FULL_HOUSE      # Treat as FH
        }
        return mapping.get(hand_type_str, HandType.HIGH_CARD)

    # ===== SCORING =====


    def calculate_score(self, cards: List[Card], game_state: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Calculate score using ScoreEngine and joker effects"""
        if game_state is None:
            game_state = self._create_game_state()
        
        # Evaluate hand type
        hand_result = self.evaluate_hand(cards)
        hand_type_str = hand_result['top']
        hand_type_enum = self._hand_type_to_enum(hand_type_str)
        scoring_cards = hand_result[hand_type_str][0] if hand_result[hand_type_str] else []
        
        # Get card IDs for scoring
        card_ids = [self._card_to_id(card) for card in scoring_cards]
        
        # Get base score from engine
        level = 0  # Bronze level
        base_score = self.score_engine.score(card_ids, hand_type_enum, level)
        
        print(f"DEBUG: Hand type: {hand_type_str} -> enum {hand_type_enum}")
        print(f"DEBUG: Scoring cards: {len(scoring_cards)}")
        print(f"DEBUG: Base score from engine: {base_score}")
        
        # Get base hand values
        hand_base_values = {
            'High Card': {'chips': 5, 'mult': 1},
            'Pair': {'chips': 10, 'mult': 2},
            'Two Pair': {'chips': 20, 'mult': 2},
            'Three of a Kind': {'chips': 30, 'mult': 3},
            'Straight': {'chips': 30, 'mult': 4},
            'Flush': {'chips': 35, 'mult': 4},
            'Full House': {'chips': 40, 'mult': 4},
            'Four of a Kind': {'chips': 60, 'mult': 7},
            'Straight Flush': {'chips': 100, 'mult': 8},
            'Five of a Kind': {'chips': 120, 'mult': 12},
            'Flush House': {'chips': 140, 'mult': 14},
            'Flush Five': {'chips': 160, 'mult': 16}
        }
        
        hand_values = hand_base_values.get(hand_type_str, {'chips': 5, 'mult': 1})
        base_chips = hand_values['chips']
        base_mult = hand_values['mult']
        
        # Add card face values to chips
        for card in scoring_cards:
            base_chips += card.base_value
        
        print(f"DEBUG: Base chips (hand + cards): {base_chips}")
        print(f"DEBUG: Base mult: {base_mult}")
        
        # Track multipliers separately
        total_add_mult = 0
        total_mult_mult = 1.0
        
        # Apply card enhancements/editions/seals
        for card in scoring_cards:
            # Enhancement effects
            if card.enhancement:
                if card.enhancement == 'bonus':
                    base_chips += 30
                elif card.enhancement == 'mult':
                    total_add_mult += 4
                elif card.enhancement == 'glass':
                    total_mult_mult *= 2.0
                    if random.random() < 0.25:
                        print(f"DEBUG: Glass card shattered!")
                elif card.enhancement == 'steel':
                    total_mult_mult *= 1.5
                elif card.enhancement == 'stone':
                    base_chips += 50
                elif card.enhancement == 'gold':
                    game_state['money'] += 3
                elif card.enhancement == 'lucky' and random.random() < 0.2:
                    game_state['money'] += 1
                    
            # Edition effects
            if card.edition:
                if card.edition == 'foil':
                    base_chips += 50
                elif card.edition == 'holographic':
                    total_add_mult += 10
                elif card.edition == 'polychrome':
                    total_mult_mult *= 1.5
                    
            # Seal effects
            if card.seal and card.seal == 'gold':
                game_state['money'] += 3
        
        print(f"DEBUG: After card effects - Chips: {base_chips}, Add Mult: {total_add_mult}, Mult Mult: {total_mult_mult}")
        
        # Apply joker effects
        for joker_id in self.player_state.jokers:
            joker_info = self.joker_id_to_info[joker_id]
            joker_obj = type('obj', (object,), {'name': joker_info.name})()
            
            # 1. Before scoring phase
            before_context = {
                'phase': 'before_scoring',
                'cards': cards,
                'scoring_cards': scoring_cards,
                'hand_type': hand_type_str
            }
            before_effect = self.joker_effects.apply_joker_effect(joker_obj, before_context, game_state)
            if before_effect and 'message' in before_effect:
                print(f"DEBUG: {joker_info.name} before scoring: {before_effect.get('message')}")
            
            # 2. Individual card scoring phase
            for card in scoring_cards:
                individual_context = {
                    'phase': 'individual_scoring',
                    'card': card,
                    'cards': cards,
                    'scoring_cards': scoring_cards,
                    'hand_type': hand_type_str
                }
                individual_effect = self.joker_effects.apply_joker_effect(joker_obj, individual_context, game_state)
                if individual_effect:
                    base_chips += individual_effect.get('chips', 0)
                    total_add_mult += individual_effect.get('mult', 0)
                    if 'x_mult' in individual_effect:
                        total_mult_mult *= individual_effect['x_mult']
                    game_state['money'] += individual_effect.get('money', 0)
                    if individual_effect.get('chips', 0) > 0 or individual_effect.get('mult', 0) > 0:
                        print(f"DEBUG: {joker_info.name} on {card.rank} of {card.suit}: +{individual_effect.get('chips', 0)} chips, +{individual_effect.get('mult', 0)} mult")
            
            # 3. Main scoring phase
            main_context = {
                'phase': 'scoring',
                'cards': cards,
                'scoring_cards': scoring_cards,
                'hand_type': hand_type_str
            }
            main_effect = self.joker_effects.apply_joker_effect(joker_obj, main_context, game_state)
            if main_effect:
                print(f"DEBUG: Joker {joker_info.name} scoring effect: {main_effect}")
                base_chips += main_effect.get('chips', 0)
                total_add_mult += main_effect.get('mult', 0)
                if 'x_mult' in main_effect:
                    total_mult_mult *= main_effect['x_mult']
                game_state['money'] += main_effect.get('money', 0)
        
        # Calculate final score
        final_mult = (base_mult + total_add_mult) * total_mult_mult
        final_score = int(base_chips * final_mult)
        
        print(f"DEBUG: Final calculation: {base_chips} chips × {final_mult} mult = {final_score}")
        
        return final_score, game_state
    def _create_game_state(self) -> Dict:
        """Create game state dict"""
        return {
            'money': self.player_state.chips,
            'deck': [self._id_to_card(cid) for cid in self.player_state.deck],
            'consumables': [],
            'jokers': [self.joker_id_to_info[jid].name for jid in self.player_state.jokers],
            'ante': self.current_ante,
            'hands_played': self.hands_played,
            'discards_remaining': self.discards_remaining
        }

    # ===== CONSUMABLES =====
    def apply_planet_card(self, planet_name: str):
        """Apply planet using the scoring engine"""
        try:
            planet_enum = Planet[planet_name.upper()]
            self.score_engine.apply_consumable(planet_enum)
        except KeyError:
            print(f"Unknown planet: {planet_name}")
    
    def apply_tarot_card(self, tarot_name: str, target_cards: List[Card], game_state: Dict):
        """Apply tarot card effects"""
        effect = self.tarots.get(tarot_name)
        
        if tarot_name == 'The Magician':
            for card in target_cards[:2]:
                card.enhancement = 'lucky'
                
        elif tarot_name == 'The Empress':
            for card in target_cards[:2]:
                card.enhancement = 'mult'
                
        elif tarot_name == 'The Hierophant':
            for card in target_cards[:2]:
                card.enhancement = 'bonus'
                
        elif tarot_name == 'The Lovers':
            if target_cards:
                target_cards[0].enhancement = 'wild'
                
        elif tarot_name == 'The Chariot':
            if target_cards:
                target_cards[0].enhancement = 'steel'
                
        elif tarot_name == 'Justice':
            if target_cards:
                target_cards[0].enhancement = 'glass'
                
        elif tarot_name == 'The Devil':
            if target_cards:
                target_cards[0].enhancement = 'gold'
                
        elif tarot_name == 'The Tower':
            if target_cards:
                target_cards[0].enhancement = 'stone'
                
        elif tarot_name == 'Strength':
            for card in target_cards[:2]:
                card.rank = min(14, card.rank + 1)
                
        elif tarot_name == 'The Hermit':
            game_state['money'] += len(game_state.get('hand', []))
            
        elif tarot_name == 'Wheel of Fortune':
            if random.random() < 0.25 and target_cards:
                card = random.choice(target_cards)
                card.edition = random.choice(['foil', 'holographic', 'polychrome'])
                
        elif tarot_name == 'The Star':
            for card in target_cards[:3]:
                card.suit = 'Diamonds'
                
        elif tarot_name == 'The Moon':
            for card in target_cards[:3]:
                card.suit = 'Clubs'
                
        elif tarot_name == 'The Sun':
            for card in target_cards[:3]:
                card.suit = 'Hearts'
                
        elif tarot_name == 'The World':
            for card in target_cards[:3]:
                card.suit = 'Spades'
        
        return game_state

    # ===== SHOP INTEGRATION =====
    def run_shop_phase(self, ante: int) -> Dict:
        """Run shop phase using the shop module"""
        shop = Shop(ante, self.player_state)
        shop_actions = []
        
        while True:
            shop_obs = shop.get_observation()
            
            # This would be replaced with actual player/agent input
            action = self._get_shop_action(shop_obs)
            
            reward, done, info = shop.step(action)
            shop_actions.append((action, reward, info))
            
            if done:
                break
                
            if "error" in info:
                print(f"Shop error: {info['error']}")
                
        return {"actions": shop_actions}
    
    def _get_shop_action(self, shop_obs: Dict) -> int:
        """Placeholder for getting shop action - would be from player/agent"""
        # For now, just skip shop
        return ShopAction.SKIP

    # ===== GAME LOOP =====
    def play_ante(self, ante_num: int) -> bool:
        """Play one ante"""
        self.current_ante = ante_num
        blinds = ['small', 'big', 'boss']
        
        for blind_type in blinds:
            # Calculate blind target
            target_score = self._get_blind_target(blind_type, ante_num)
            
            # Play the blind
            score = self._play_blind(target_score)
            
            if score >= target_score:
                # Award money
                reward = self._calculate_blind_reward(blind_type, ante_num)
                self.player_state.chips += reward
                
                # Shop phase (except after ante 8 boss)
                if not (ante_num == 8 and blind_type == 'boss'):
                    self.run_shop_phase(ante_num)
            else:
                return False  # Game over
                
        return True  # Ante completed
    
    def _get_blind_target(self, blind_type: str, ante: int) -> int:
        """Get score target for blind"""
        base_scores = {
            'small': 100,
            'big': 200,
            'boss': 350
        }
        multiplier = 1.5 ** (ante - 1)
        return int(base_scores[blind_type] * multiplier)
    
    def _calculate_blind_reward(self, blind_type: str, ante: int) -> int:
        """Calculate money reward for beating blind"""
        base_rewards = {
            'small': 3,
            'big': 4,
            'boss': 5
        }
        return base_rewards[blind_type] + ante
    
    def _play_blind(self, target_score: int) -> int:
        """Placeholder for playing a blind - would involve actual gameplay"""
        # For now, return a random score
        return random.randint(50, 500)

    # ===== DISCARD EFFECTS =====
    def apply_joker_discard_effects(self, discarded_cards: List[Card], game_state: Dict):
        """Handle joker effects when discarding"""
        for joker_id in self.player_state.jokers:
            joker_info = self.joker_id_to_info[joker_id]
            
            # Create joker object with name attribute
            joker_obj = type('obj', (object,), {'name': joker_info.name})()
            
            context = {
                'phase': 'discard',
                'discarded_cards': discarded_cards,
                'last_discarded_card': discarded_cards[-1] if discarded_cards else None
            }
            
            effect = self.joker_effects.apply_joker_effect(
                joker_obj,
                context,
                game_state
            )
            
            if effect:
                game_state['money'] += effect.get('money', 0)
    # ===== END ROUND EFFECTS =====
    def apply_joker_end_round_effects(self, game_state: Dict) -> List[Dict]:
        """Handle end-of-round joker effects"""
        effects = []
        
        for joker_id in self.player_state.jokers:
            joker_info = self.joker_id_to_info[joker_id]
            
            # Create joker object with name attribute
            joker_obj = type('obj', (object,), {'name': joker_info.name})()
            
            context = {'phase': 'end_round'}
            effect = self.joker_effects.apply_joker_effect(
                joker_obj,
                context,
                game_state
            )
            
            if effect:
                effects.append(effect)
        
        # Handle destruction effects
        destruction_effects = self.joker_effects.end_of_round_effects(game_state)
        effects.extend(destruction_effects)
        
        return effects
    # ===== UTILITY METHODS =====
    def get_joker_info(self, joker_name: str) -> Dict:
        """Get information about a specific joker"""
        joker_id = self.joker_name_to_id.get(joker_name)
        if joker_id:
            return self.joker_id_to_info[joker_id].__dict__
        return {}
    
    def get_all_joker_names(self) -> List[str]:
        """Get list of all implemented joker names"""
        return list(self.joker_name_to_id.keys())


# Usage and testing
if __name__ == "__main__":
    simulator = BalatroSimulator()
    print(f"Total jokers available: {len(simulator.get_all_joker_names())}")
    print("\nFirst 10 jokers:")
    for joker_name in list(simulator.get_all_joker_names())[:10]:
        info = simulator.get_joker_info(joker_name)
        print(f"  {joker_name}: Cost ${info['base_cost']} - {info['effect']}")
