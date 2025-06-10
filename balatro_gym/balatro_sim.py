import random
import copy
from complete_joker_effects import CompleteJokerEffects

class BalatroSimulator:
    def __init__(self):
        # Planet cards (hand upgrades)
        self.planets = {
            'Mercury': 'Pair',
            'Venus': 'Two Pair', 
            'Earth': 'Full House',
            'Mars': 'Four of a Kind',
            'Jupiter': 'Flush',
            'Saturn': 'Three of a Kind',
            'Uranus': 'Straight',
            'Neptune': 'Straight Flush',
            'Pluto': 'High Card',
            'Planet X': 'Five of a Kind',
            'Ceres': 'Flush House',
            'Eris': 'Flush Five'
        }
        
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
        
        # Complete hand values (EXACT from Balatro)
        self.base_hands = {
            'Flush Five': {'chips': 160, 'mult': 16, 'l_chips': 50, 'l_mult': 3},
            'Flush House': {'chips': 140, 'mult': 14, 'l_chips': 40, 'l_mult': 4},
            'Five of a Kind': {'chips': 120, 'mult': 12, 'l_chips': 35, 'l_mult': 3},
            'Straight Flush': {'chips': 100, 'mult': 8, 'l_chips': 40, 'l_mult': 4},
            'Four of a Kind': {'chips': 60, 'mult': 7, 'l_chips': 30, 'l_mult': 3},
            'Full House': {'chips': 40, 'mult': 4, 'l_chips': 25, 'l_mult': 2},
            'Flush': {'chips': 35, 'mult': 4, 'l_chips': 15, 'l_mult': 2},
            'Straight': {'chips': 30, 'mult': 4, 'l_chips': 30, 'l_mult': 3},
            'Three of a Kind': {'chips': 30, 'mult': 3, 'l_chips': 20, 'l_mult': 2},
            'Two Pair': {'chips': 20, 'mult': 2, 'l_chips': 20, 'l_mult': 1},
            'Pair': {'chips': 10, 'mult': 2, 'l_chips': 15, 'l_mult': 1},
            'High Card': {'chips': 5, 'mult': 1, 'l_chips': 10, 'l_mult': 1}
        }
        
        # Initialize the complete joker effects system
        self.joker_effects = CompleteJokerEffects()
        
        # Complete joker database from extracted code
        self.jokers = {
            # ===== BASIC JOKERS =====
            'Joker': {'type': 'basic', 'effect': 'mult', 'value': 4},
            'Greedy Joker': {'type': 'suit', 'suit': 'Diamonds', 'mult': 3},
            'Lusty Joker': {'type': 'suit', 'suit': 'Hearts', 'mult': 3},
            'Wrathful Joker': {'type': 'suit', 'suit': 'Spades', 'mult': 3},
            'Gluttonous Joker': {'type': 'suit', 'suit': 'Clubs', 'mult': 3},
            
            # ===== HAND-TYPE JOKERS =====
            'Jolly Joker': {'type': 'hand_type', 'hand': 'Pair', 'mult': 8},
            'Zany Joker': {'type': 'hand_type', 'hand': 'Three of a Kind', 'mult': 12},
            'Mad Joker': {'type': 'hand_type', 'hand': 'Two Pair', 'mult': 10},
            'Crazy Joker': {'type': 'hand_type', 'hand': 'Straight', 'mult': 12},
            'Droll Joker': {'type': 'hand_type', 'hand': 'Flush', 'mult': 10},
            'Sly Joker': {'type': 'hand_type', 'hand': 'Pair', 'chips': 50},
            'Wily Joker': {'type': 'hand_type', 'hand': 'Three of a Kind', 'chips': 100},
            'Clever Joker': {'type': 'hand_type', 'hand': 'Two Pair', 'chips': 80},
            'Devious Joker': {'type': 'hand_type', 'hand': 'Straight', 'chips': 100},
            'Crafty Joker': {'type': 'hand_type', 'hand': 'Flush', 'chips': 80},
            
            # ===== INDIVIDUAL CARD EFFECTS =====
            'Fibonacci': {'type': 'rank', 'ranks': [2, 3, 5, 8, 14], 'mult': 8},
            'Scholar': {'type': 'rank', 'ranks': [14], 'chips': 20, 'mult': 4},
            'Even Steven': {'type': 'even_odd', 'parity': 'even', 'mult': 4},
            'Odd Todd': {'type': 'even_odd', 'parity': 'odd', 'chips': 31},
            'Scary Face': {'type': 'face', 'chips': 30},
            'Smiley Face': {'type': 'face', 'mult': 5},
            'Walkie Talkie': {'type': 'rank', 'ranks': [4, 10], 'chips': 10, 'mult': 4},
            'Triboulet': {'type': 'rank', 'ranks': [12, 13], 'x_mult': 2},
            'Ancient Joker': {'type': 'special', 'effect': 'ancient_suit'},
            'The Idol': {'type': 'special', 'effect': 'idol_card'},
            'Photograph': {'type': 'special', 'effect': 'first_face'},
            'Golden Ticket': {'type': 'enhancement', 'target': 'gold', 'money': 4},
            'Rough Gem': {'type': 'suit', 'suit': 'Diamonds', 'money': 1},
            'Arrowhead': {'type': 'suit', 'suit': 'Spades', 'chips': 50},
            'Onyx Agate': {'type': 'suit', 'suit': 'Clubs', 'mult': 7},
            'Bloodstone': {'type': 'suit', 'suit': 'Hearts', 'x_mult': 2, 'chance': 0.5},
            
            # ===== GROWING JOKERS =====
            'Ride the Bus': {'type': 'growing', 'effect': 'no_face_mult'},
            'Green Joker': {'type': 'growing', 'effect': 'hand_discard_mult'},
            'Red Card': {'type': 'growing', 'effect': 'skip_mult'},
            'Spare Trousers': {'type': 'growing', 'effect': 'two_pair_full_house'},
            'Square Joker': {'type': 'growing', 'effect': 'four_cards'},
            'Runner': {'type': 'growing', 'effect': 'straight_chips'},
            'Flash Card': {'type': 'growing', 'effect': 'reroll_mult'},
            'Wee Joker': {'type': 'growing', 'effect': 'twos_chips'},
            'Ceremonial Dagger': {'type': 'growing', 'effect': 'destroy_joker'},
            'Swashbuckler': {'type': 'growing', 'effect': 'sell_joker'},
            
            # ===== CONDITIONAL JOKERS =====
            'Half Joker': {'type': 'conditional', 'condition': 'hand_size_3', 'mult': 20},
            'Abstract Joker': {'type': 'conditional', 'condition': 'joker_count', 'mult_per': 3},
            'Supernova': {'type': 'conditional', 'condition': 'hand_played_count'},
            'Blue Joker': {'type': 'conditional', 'condition': 'deck_size', 'chips_per': 2},
            'Banner': {'type': 'conditional', 'condition': 'discards_left', 'chips_per': 30},
            'Mystic Summit': {'type': 'conditional', 'condition': 'no_discards', 'mult': 15},
            'Acrobat': {'type': 'conditional', 'condition': 'final_hand', 'x_mult': 3},
            'Bull': {'type': 'conditional', 'condition': 'money', 'chips_per': 2},
            'Bootstraps': {'type': 'conditional', 'condition': 'money_5', 'mult_per': 2},
            'Seeing Double': {'type': 'conditional', 'condition': 'clubs_other_suit', 'x_mult': 2},
            'Flower Pot': {'type': 'conditional', 'condition': 'all_suits', 'x_mult': 3},
            'Blackboard': {'type': 'conditional', 'condition': 'all_black_suits', 'x_mult': 3},
            'Joker Stencil': {'type': 'conditional', 'condition': 'empty_joker_slots'},
            'Driver\'s License': {'type': 'conditional', 'condition': '16_enhanced', 'x_mult': 3},
            
            # ===== UTILITY JOKERS =====
            'Blueprint': {'type': 'utility', 'effect': 'copy_right'},
            'Brainstorm': {'type': 'utility', 'effect': 'copy_leftmost'},
            'Four Fingers': {'type': 'utility', 'effect': 'four_card_flush_straight'},
            'Shortcut': {'type': 'utility', 'effect': 'skip_straight'},
            'Smeared Joker': {'type': 'utility', 'effect': 'hearts_diamonds_same'},
            'Pareidolia': {'type': 'utility', 'effect': 'all_face_cards'},
            'Mime': {'type': 'utility', 'effect': 'retrigger_hand'},
            'Sock and Buskin': {'type': 'utility', 'effect': 'retrigger_faces'},
            'Hanging Chad': {'type': 'utility', 'effect': 'retrigger_first'},
            'Hack': {'type': 'utility', 'effect': 'retrigger_2345'},
            'Dusk': {'type': 'utility', 'effect': 'retrigger_final'},
            
            # ===== ECONOMIC JOKERS =====
            'Trading Card': {'type': 'economic', 'effect': 'first_discard_single'},
            'Business Card': {'type': 'economic', 'effect': 'face_money_chance'},
            'Rough Gem': {'type': 'economic', 'effect': 'diamond_money'},
            'Reserved Parking': {'type': 'economic', 'effect': 'face_money_chance_hand'},
            'Mail-In Rebate': {'type': 'economic', 'effect': 'discard_rank_money'},
            'Delayed Gratification': {'type': 'economic', 'effect': 'discard_money'},
            'To Do List': {'type': 'economic', 'effect': 'specific_hand_money'},
            'Faceless Joker': {'type': 'economic', 'effect': 'face_discard_money'},
            'Matador': {'type': 'economic', 'effect': 'boss_ability_money'},
            'Vagabond': {'type': 'economic', 'effect': 'poor_tarot'},
            
            # ===== DESTRUCTION/RISK JOKERS =====
            'Popcorn': {'type': 'destruction', 'mult': 20, 'decay': 4},
            'Turtle Bean': {'type': 'destruction', 'hand_size': -1, 'decay': 1},
            'Ice Cream': {'type': 'destruction', 'chips': 100, 'decay': 10},
            'Ramen': {'type': 'destruction', 'x_mult': 2.0, 'decay': 0.01},
            'Gros Michel': {'type': 'destruction', 'mult': 15, 'destroy_chance': 1/6},
            'Glass Joker': {'type': 'destruction', 'effect': 'glass_destroy_mult'},
            'Caino': {'type': 'destruction', 'effect': 'face_destroy_mult'},
            
            # ===== LEGENDARY JOKERS =====
            'The Family': {'type': 'legendary', 'hand': 'Four of a Kind', 'x_mult': 4},
            'The Order': {'type': 'legendary', 'hand': 'Straight', 'x_mult': 3},
            'The Tribe': {'type': 'legendary', 'hand': 'Flush', 'x_mult': 2},
            'Chicot': {'type': 'legendary', 'effect': 'disable_boss'},
            'Perkeo': {'type': 'legendary', 'effect': 'copy_consumable'},
            'Mr. Bones': {'type': 'legendary', 'effect': 'save_25_percent'},
            
            # ===== SPECIAL JOKERS =====
            'Invisible Joker': {'type': 'special', 'effect': 'copy_after_rounds'},
            'Certificate': {'type': 'special', 'effect': 'add_enhanced_card'},
            'DNA': {'type': 'special', 'effect': 'copy_single_card'},
            'Marble Joker': {'type': 'special', 'effect': 'add_stone_card'},
            'Vampire': {'type': 'special', 'effect': 'destroy_enhanced'},
            'Midas Mask': {'type': 'special', 'effect': 'faces_to_gold'},
            'Space Joker': {'type': 'special', 'effect': 'level_up_chance'},
            'Obelisk': {'type': 'special', 'effect': 'most_played_hand'},
            'Hologram': {'type': 'special', 'effect': 'playing_card_added'},
            'Throwback': {'type': 'special', 'effect': 'skip_blind'},
            'Campfire': {'type': 'special', 'effect': 'sell_card_mult'},
            'Rocket': {'type': 'special', 'effect': 'boss_money_grow'},
            'Satellite': {'type': 'special', 'effect': 'planet_money'},
            'Constellation': {'type': 'special', 'effect': 'planet_mult'},
            'Seance': {'type': 'special', 'effect': 'straight_flush_spectral'},
            'Superposition': {'type': 'special', 'effect': 'ace_straight_tarot'},
            'Sixth Sense': {'type': 'special', 'effect': 'destroy_6_spectral'},
            
            # ===== MORE JOKERS FROM EXTRACTED CODE =====
            'Misprint': {'type': 'random', 'mult_min': 0, 'mult_max': 23},
            'Stuntman': {'type': 'basic', 'chips': 250, 'hand_size': -1},
            'Loyalty Card': {'type': 'special', 'effect': 'every_5th_hand', 'x_mult': 4},
            'Card Sharp': {'type': 'conditional', 'condition': 'hand_played_twice', 'x_mult': 3},
            'Cavendish': {'type': 'basic', 'x_mult': 3},
            'Stone Joker': {'type': 'conditional', 'condition': 'stone_cards', 'chips_per': 25},
            'Steel Joker': {'type': 'conditional', 'condition': 'steel_cards', 'x_mult_per': 0.2},
            'Raised Fist': {'type': 'special', 'effect': 'lowest_rank_mult'},
            'Chaos the Clown': {'type': 'utility', 'effect': 'free_rerolls'},
            'Credit Card': {'type': 'special', 'effect': 'debt_money'},
            'Ceremonial Dagger': {'type': 'growing', 'effect': 'destroy_right_joker'},
            'Marble Joker': {'type': 'special', 'effect': 'add_stone_card_to_deck'},
            '8 Ball': {'type': 'special', 'effect': 'create_tarot_on_8', 'chance': 0.25},
            'Dusk': {'type': 'utility', 'effect': 'retrigger_final_hand'},
            'Pareidolia': {'type': 'utility', 'effect': 'all_face_cards'},
            'Hack': {'type': 'utility', 'effect': 'retrigger_low_cards'},
            'Delayed Gratification': {'type': 'economic', 'effect': 'discard_money'},
            'Baron': {'type': 'conditional', 'condition': 'king_in_hand', 'x_mult': 1.5},
            'Shoot the Moon': {'type': 'conditional', 'condition': 'queen_in_hand', 'hand_mult': 13},
            'Reserved Parking': {'type': 'economic', 'effect': 'face_hand_money'},
            'Mail-In Rebate': {'type': 'economic', 'effect': 'discard_rank_money'},
            'Hit the Road': {'type': 'growing', 'effect': 'jack_discard_mult'},
            'Faceless Joker': {'type': 'economic', 'effect': 'face_discard_money'},
            'Yorick': {'type': 'growing', 'effect': 'discard_counter_mult'},
            'Castle': {'type': 'growing', 'effect': 'suit_discard_chips'},
            'Satellite': {'type': 'economic', 'effect': 'planet_money'},
            'Rocket': {'type': 'growing', 'effect': 'boss_money_growth'},
            'Oops! All 6s': {'type': 'special', 'effect': 'double_probabilities'},
            'Burglar': {'type': 'special', 'effect': 'trade_discards_hands'},
            'Blackboard': {'type': 'conditional', 'condition': 'all_black_hand', 'x_mult': 3},
            'Runner': {'type': 'growing', 'effect': 'straight_chip_growth'},
            'Ice Cream': {'type': 'decaying', 'chips': 100, 'decay_per_hand': 10},
            'Seltzer': {'type': 'decaying', 'retriggers': 1, 'uses': 10},
            'Popcorn': {'type': 'decaying', 'mult': 20, 'decay_per_round': 4},
            'Turtle Bean': {'type': 'decaying', 'hand_size': -1, 'decay_per_round': 1},
            'Ramen': {'type': 'decaying', 'x_mult': 2.0, 'decay_per_discard': 0.01},
            'Gros Michel': {'type': 'risky', 'mult': 15, 'destroy_chance': 1/6},
            'Mr. Bones': {'type': 'safety', 'effect': 'save_at_25_percent'},
            'Luchador': {'type': 'special', 'effect': 'disable_boss_when_sold'},
            'Chicot': {'type': 'legendary', 'effect': 'disable_boss_blind'},
            'Madness': {'type': 'growing', 'effect': 'destroy_joker_for_mult'},
            'Riff-raff': {'type': 'special', 'effect': 'create_common_jokers'},
            'Cartomancer': {'type': 'special', 'effect': 'create_tarot_blind'},
            'Perkeo': {'type': 'legendary', 'effect': 'copy_random_consumable'},
            'Certificate': {'type': 'special', 'effect': 'add_random_enhanced_card'},
            'DNA': {'type': 'special', 'effect': 'copy_single_card_first_hand'},
            'Sixth Sense': {'type': 'special', 'effect': 'destroy_6_for_spectral'},
            'Seance': {'type': 'consumable', 'effect': 'straight_flush_spectral'},
            'Superposition': {'type': 'consumable', 'effect': 'ace_straight_tarot'},
            'Vagabond': {'type': 'economic', 'effect': 'poor_tarot_creation'},
            'Constellation': {'type': 'growing', 'effect': 'planet_mult_growth'},
            'Fortune Teller': {'type': 'conditional', 'effect': 'tarot_usage_mult'},
            'Matador': {'type': 'economic', 'effect': 'boss_ability_money'},
        }

    def get_x_same(self, num, hand):
        """Find cards of the same rank (from Balatro source)"""
        vals = [[] for _ in range(15)]  # 14 ranks + 1 for indexing
        
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

    def get_flush(self, hand, four_fingers=False):
        """Find flush (from Balatro source)"""
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

    def get_straight(self, hand, four_fingers=False, shortcut=False):
        """Find straight (from Balatro source)"""
        ret = []
        required_cards = 4 if four_fingers else 5
        
        if len(hand) > 5 or len(hand) < required_cards:
            return ret
            
        t = []
        ids = {}
        
        for card in hand:
            rank = card.rank
            if 1 < rank < 15:  # Valid ranks
                if rank in ids:
                    ids[rank].append(card)
                else:
                    ids[rank] = [card]
        
        straight_length = 0
        straight = False
        can_skip = shortcut
        skipped_rank = False
        
        for i in range(14, 1, -1):  # Check from Ace down to 2
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
            
            for rank in [14, 2, 3, 4, 5]:  # A-2-3-4-5
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
            ret.append(t[:required_cards])  # Only return required number of cards
            
        return ret

    def get_highest(self, hand):
        """Get highest cards for high card hand"""
        return [hand]  # Return all cards for high card evaluation

    def evaluate_hand(self, cards):
        """Use EXACT logic from poker_evaluation.txt"""
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

        # Get all the parts using helper functions
        parts = {
            "_5": self.get_x_same(5, cards),
            "_4": self.get_x_same(4, cards),
            "_3": self.get_x_same(3, cards),
            "_2": self.get_x_same(2, cards),
            "_flush": self.get_flush(cards),
            "_straight": self.get_straight(cards),
            "_highest": self.get_highest(cards)
        }

        # Evaluate hands in priority order (from Balatro source)
        
        # Flush Five (5 of a kind + flush)
        if parts["_5"] and parts["_flush"]:
            results["Flush Five"] = parts["_5"]
            if not results["top"]:
                results["top"] = "Flush Five"

        # Flush House (full house + flush)
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
            
            # Add flush cards
            for card in _f[0]:
                ret.append(card)
            
            # Add straight cards not in flush
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

        # Cascade lower hands (from Balatro source)
        if results["Five of a Kind"]:
            results["Four of a Kind"] = [results["Five of a Kind"][0][:4]]

        if results["Four of a Kind"]:
            results["Three of a Kind"] = [results["Four of a Kind"][0][:3]]

        if results["Three of a Kind"]:
            results["Pair"] = [results["Three of a Kind"][0][:2]]

        return results

    def apply_planet_card(self, planet_name, hand_levels):
        """Apply planet card to upgrade hand level"""
        hand_type = self.planets[planet_name]
        hand_levels[hand_type] = hand_levels.get(hand_type, 1) + 1
        return hand_levels
    
    def apply_tarot_card(self, tarot_name, target_cards, game_state):
        """Apply tarot card effects"""
        effect = self.tarots[tarot_name]
        
        if tarot_name == 'The Magician':
            # Enhance up to 2 cards to Lucky
            for i, card in enumerate(target_cards[:2]):
                card.enhancement = 'lucky'
                
        elif tarot_name == 'The Empress':
            # Enhance up to 2 cards to Mult
            for i, card in enumerate(target_cards[:2]):
                card.enhancement = 'mult'
                
        elif tarot_name == 'The Emperor':
            # Create 2-4 Tarot cards
            game_state['consumables'].extend(['random_tarot'] * 3)
            
        elif tarot_name == 'The Hierophant':
            # Enhance up to 2 cards to Bonus
            for i, card in enumerate(target_cards[:2]):
                card.enhancement = 'bonus'
                
        elif tarot_name == 'The Lovers':
            # Enhance 1 card to Wild
            if target_cards:
                target_cards[0].enhancement = 'wild'
                
        elif tarot_name == 'The Chariot':
            # Enhance 1 card to Steel
            if target_cards:
                target_cards[0].enhancement = 'steel'
                
        elif tarot_name == 'Strength':
            # Increase rank of up to 2 cards by 1
            for card in target_cards[:2]:
                card.rank = min(14, card.rank + 1)
                
        elif tarot_name == 'The Hermit':
            # Gain money equal to hand size
            game_state['money'] += len(game_state['hand'])
            
        elif tarot_name == 'Wheel of Fortune':
            # 1 in 4 chance to add foil, holo, or poly to random card
            if random.random() < 0.25:
                card = random.choice(target_cards)
                card.edition = random.choice(['foil', 'holographic', 'polychrome'])
                
        elif tarot_name == 'Justice':
            # Enhance 1 card to Glass
            if target_cards:
                target_cards[0].enhancement = 'glass'
                
        elif tarot_name == 'The Hanged Man':
            # Destroy up to 2 cards, gain money
            for card in target_cards[:2]:
                game_state['money'] += 2
                # Remove card from deck
                
        elif tarot_name == 'Death':
            # Convert leftmost 2 cards to rightmost 2 cards
            if len(target_cards) >= 4:
                for i in range(2):
                    target_cards[i].rank = target_cards[-(i+1)].rank
                    target_cards[i].suit = target_cards[-(i+1)].suit
                    
        elif tarot_name == 'Temperance':
            # Gain money equal to Joker sell values
            for joker in game_state['jokers']:
                game_state['money'] += joker.sell_value
                
        elif tarot_name == 'The Devil':
            # Enhance 1 card to Gold
            if target_cards:
                target_cards[0].enhancement = 'gold'
                
        elif tarot_name == 'The Tower':
            # Enhance 1 card to Stone
            if target_cards:
                target_cards[0].enhancement = 'stone'
                
        elif tarot_name == 'The Star':
            # Convert up to 3 cards to Diamonds
            for card in target_cards[:3]:
                card.suit = 'Diamonds'
                
        elif tarot_name == 'The Moon':
            # Convert up to 3 cards to Clubs  
            for card in target_cards[:3]:
                card.suit = 'Clubs'
                
        elif tarot_name == 'The Sun':
            # Convert up to 3 cards to Hearts
            for card in target_cards[:3]:
                card.suit = 'Hearts'
                
        elif tarot_name == 'Judgement':
            # Create random Planet card
            game_state['consumables'].append(random.choice(list(self.planets.keys())))
            
        elif tarot_name == 'The World':
            # Convert up to 3 cards to Spades
            for card in target_cards[:3]:
                card.suit = 'Spades'
        
        return game_state
    
    def calculate_card_effects(self, card, context):
        """Calculate individual card enhancement/edition/seal effects"""
        effect = {'chips': 0, 'mult': 0, 'x_mult': 1, 'money': 0, 'retriggers': 0}
        
        # Enhancement effects
        if hasattr(card, 'enhancement') and card.enhancement:
            enh = self.enhancements.get(card.enhancement, {})
            effect['chips'] += enh.get('chips', 0)
            effect['mult'] += enh.get('mult', 0)
            effect['x_mult'] *= enh.get('x_mult', 1)
            
            if card.enhancement == 'glass' and context == 'scoring':
                # Glass card X2 mult but 1/4 chance to destroy
                if random.random() < 0.25:
                    effect['destroy'] = True
                    
            elif card.enhancement == 'gold' and context == 'scoring':
                effect['money'] += 3
                
            elif card.enhancement == 'lucky' and context == 'scoring':
                if random.random() < 0.2:
                    effect['money'] += 1
                    
        # Edition effects  
        if hasattr(card, 'edition') and card.edition:
            ed = self.editions.get(card.edition, {})
            effect['chips'] += ed.get('chips', 0)
            effect['mult'] += ed.get('mult', 0)
            effect['x_mult'] *= ed.get('x_mult', 1)
            
        # Seal effects
        if hasattr(card, 'seal') and card.seal:
            seal = self.seals.get(card.seal, {})
            if card.seal == 'red':
                effect['retriggers'] += 1
            elif card.seal == 'gold' and context == 'scoring':
                effect['money'] += 3
                
        return effect
    
    def calculate_steel_joker_effect(self, steel_cards_count):
        """Calculate Steel Joker effect based on steel cards in deck"""
        if steel_cards_count > 0:
            return {'x_mult': 1 + (0.2 * steel_cards_count)}
        return None
    
    def calculate_stone_joker_effect(self, stone_cards_count):
        """Calculate Stone Joker effect based on stone cards in deck"""
        if stone_cards_count > 0:
            return {'chips': 25 * stone_cards_count}
        return None

    def calculate_score(self, cards, jokers, hand_levels, game_state=None):
        """Updated scoring with complete joker system"""
        if game_state is None:
            game_state = {'money': 0, 'deck': [], 'consumables': [], 'jokers': jokers}
            
        # 1. Evaluate hand type
        hand_result = self.evaluate_hand(cards)
        hand_type = hand_result['top']
        scoring_cards = hand_result[hand_type][0] if hand_result[hand_type] else []
        
        # 2. Base scoring
        base = self.base_hands[hand_type]
        level = hand_levels.get(hand_type, 1)
        
        chips = base['chips'] + (level - 1) * base['l_chips']
        mult = base['mult'] + (level - 1) * base['l_mult']
        
        # 3. Card values and effects
        for card in scoring_cards:
            chips += card.base_value
            
            # Individual card enhancements
            card_effect = self.calculate_card_effects(card, 'scoring')
            chips += card_effect['chips']
            mult += card_effect['mult']
            mult *= card_effect['x_mult']
            game_state['money'] += card_effect['money']
        
        # 4. Apply joker effects using complete system
        for joker in jokers:
            # Before scoring effects
            before_context = {
                'phase': 'before_scoring',
                'cards': cards,
                'scoring_cards': scoring_cards,
                'hand_type': hand_type
            }
            before_effect = self.joker_effects.apply_joker_effect(joker, before_context, game_state)
            
            # Individual card effects
            for card in scoring_cards:
                individual_context = {
                    'phase': 'individual_scoring',
                    'card': card,
                    'cards': cards,
                    'scoring_cards': scoring_cards,
                    'hand_type': hand_type
                }
                individual_effect = self.joker_effects.apply_joker_effect(joker, individual_context, game_state)
                if individual_effect:
                    chips += individual_effect.get('chips', 0)
                    mult += individual_effect.get('mult', 0)
                    if 'x_mult' in individual_effect:
                        mult *= individual_effect['x_mult']
                    game_state['money'] += individual_effect.get('money', 0)
            
            # Main scoring effects
            main_context = {
                'phase': 'scoring',
                'cards': cards,
                'scoring_cards': scoring_cards,
                'hand_type': hand_type
            }
            main_effect = self.joker_effects.apply_joker_effect(joker, main_context, game_state)
            if main_effect:
                chips += main_effect.get('chips', 0)
                mult += main_effect.get('mult', 0)
                if 'x_mult' in main_effect:
                    mult *= main_effect['x_mult']
                game_state['money'] += main_effect.get('money', 0)
        
        # 5. Handle special joker effects that depend on deck composition
        if game_state.get('deck'):
            steel_count = sum(1 for card in game_state['deck'] if getattr(card, 'enhancement', None) == 'steel')
            stone_count = sum(1 for card in game_state['deck'] if getattr(card, 'enhancement', None) == 'stone')
            
            # Steel Joker effect
            steel_joker_effect = self.calculate_steel_joker_effect(steel_count)
            if steel_joker_effect:
                mult *= steel_joker_effect['x_mult']
                
            # Stone Joker effect  
            stone_joker_effect = self.calculate_stone_joker_effect(stone_count)
            if stone_joker_effect:
                chips += stone_joker_effect['chips']
                
        return int(chips * mult), game_state
    
    def apply_joker_discard_effects(self, joker, discarded_cards, game_state):
        """Handle joker effects when discarding"""
        context = {
            'phase': 'discard',
            'discarded_cards': discarded_cards,
            'last_discarded_card': discarded_cards[-1] if discarded_cards else None
        }
        return self.joker_effects.apply_joker_effect(joker, context, game_state)
    
    def apply_joker_end_round_effects(self, jokers, game_state):
        """Handle end-of-round joker effects"""
        effects = []
        for joker in jokers:
            context = {'phase': 'end_round'}
            effect = self.joker_effects.apply_joker_effect(joker, context, game_state)
            if effect:
                effects.append(effect)
        
        # Handle destruction effects
        effects.extend(self.joker_effects.end_of_round_effects(game_state))
        return effects
    
    def get_joker_info(self, joker_name):
        """Get information about a specific joker"""
        return self.jokers.get(joker_name, {})
    
    def get_all_joker_names(self):
        """Get list of all implemented joker names"""
        return list(self.jokers.keys())

# Usage and testing
if __name__ == "__main__":
    simulator = BalatroSimulator()
    print(f"Total jokers implemented: {len(simulator.get_all_joker_names())}")
    print("\nFirst 10 jokers:")
    for joker_name in list(simulator.get_all_joker_names())[:10]:
        info = simulator.get_joker_info(joker_name)
        print(f"  {joker_name}: {info}")
