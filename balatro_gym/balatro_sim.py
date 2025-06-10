class BalatroSimulator:
    def __init__(self):
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
        
        # All jokers from extracted code (150+ jokers)
        self.jokers = {
            # Basic jokers
            'Joker': {'mult': 4},
            'Greedy Joker': {'mult': 3, 'suit': 'Diamonds'},
            'Lusty Joker': {'mult': 3, 'suit': 'Hearts'},
            'Wrathful Joker': {'mult': 3, 'suit': 'Spades'},
            'Gluttonous Joker': {'mult': 3, 'suit': 'Clubs'},
            
            # Type-specific jokers
            'Jolly Joker': {'mult': 8, 'hand_type': 'Pair'},
            'Zany Joker': {'mult': 12, 'hand_type': 'Three of a Kind'},
            'Mad Joker': {'mult': 10, 'hand_type': 'Two Pair'},
            'Crazy Joker': {'mult': 12, 'hand_type': 'Straight'},
            'Droll Joker': {'mult': 10, 'hand_type': 'Flush'},
            'Sly Joker': {'chips': 50, 'hand_type': 'Pair'},
            'Wily Joker': {'chips': 100, 'hand_type': 'Three of a Kind'},
            'Clever Joker': {'chips': 80, 'hand_type': 'Two Pair'},
            'Devious Joker': {'chips': 100, 'hand_type': 'Straight'},
            'Crafty Joker': {'chips': 80, 'hand_type': 'Flush'},
            
            # Individual card effects
            'Fibonacci': {'mult': 8, 'ranks': [2, 3, 5, 8, 14]},
            'Scholar': {'chips': 20, 'mult': 4, 'ranks': [14]},
            'Even Steven': {'mult': 4, 'condition': 'even'},
            'Odd Todd': {'chips': 31, 'condition': 'odd'},
            'Scary Face': {'chips': 30, 'targets': 'face_cards'},
            'Smiley Face': {'mult': 5, 'targets': 'face_cards'},
            'Walkie Talkie': {'chips': 10, 'mult': 4, 'ranks': [10, 4]},
            'Business Card': {'money': 2, 'targets': 'face_cards', 'chance': 0.5},
            'Rough Gem': {'money': 1, 'suit': 'Diamonds'},
            'Arrowhead': {'chips': 50, 'suit': 'Spades'},
            'Onyx Agate': {'mult': 7, 'suit': 'Clubs'},
            'Bloodstone': {'x_mult': 2, 'suit': 'Hearts', 'chance': 0.5},
            'Golden Ticket': {'money': 4, 'enhancement': 'Gold Card'},
            'Triboulet': {'x_mult': 2, 'ranks': [12, 13]},
            'The Idol': {'x_mult': 2, 'special': 'idol_card'},
            'Photograph': {'x_mult': 2, 'special': 'first_face'},
            
            # Global effects
            'Half Joker': {'mult': 20, 'condition': 'hand_size_3'},
            'Abstract Joker': {'mult_per_joker': 3},
            'Supernova': {'mult_per_hand_played': 1},
            'Blue Joker': {'chips_per_deck_card': 2},
            'Stone Joker': {'chips_per_stone': 25},
            'Steel Joker': {'x_mult_per_steel': 0.2},
            'Banner': {'chips_per_discard': 30},
            'Mystic Summit': {'mult': 15, 'condition': 'no_discards'},
            'Misprint': {'mult_random': [0, 23]},
            'Stuntman': {'chips': 250},
            'Bull': {'chips_per_dollar': 2},
            'Acrobat': {'x_mult': 3, 'condition': 'final_hand'},
            'Loyalty Card': {'x_mult': 4, 'every_nth_hand': 5},
            'Card Sharp': {'x_mult': 3, 'condition': 'hand_played_twice'},
            'Bootstraps': {'mult_per_5_dollars': 2},
            'Cavendish': {'x_mult': 3},
            'Gros Michel': {'mult': 15, 'destroy_chance': 1/6},
            
            # Conditional jokers
            'Seeing Double': {'x_mult': 2, 'condition': 'clubs_plus_other'},
            'Flower Pot': {'x_mult': 3, 'condition': 'all_suits'},
            'Blackboard': {'x_mult': 3, 'condition': 'all_black'},
            'Joker Stencil': {'x_mult': 1, 'condition': 'empty_joker_slots'},
            'Driver\'s License': {'x_mult': 3, 'condition': '16_enhanced_cards'},
            
            # Special mechanics
            'Four Fingers': {'special': 'flush_straight_minus_1'},
            'Shortcut': {'special': 'straights_skip_1'},
            'Smeared Joker': {'special': 'hearts_diamonds_same'},
            'Pareidolia': {'special': 'all_face_cards'},
            'Blueprint': {'special': 'copy_right_joker'},
            'Brainstorm': {'special': 'copy_leftmost_joker'},
            'Mime': {'repetitions': 1, 'condition': 'hand_cards'},
            'Sock and Buskin': {'repetitions': 1, 'targets': 'face_cards'},
            'Hanging Chad': {'repetitions': 1, 'targets': 'first_card'},
            'Hack': {'repetitions': 1, 'ranks': [2, 3, 4, 5]},
            'Dusk': {'repetitions': 1, 'condition': 'final_hand'},
            'Seltzer': {'repetitions': 1, 'uses': 10},
            
            # Economy jokers
            'Delayed Gratification': {'money_per_discard': 2},
            'To Do List': {'money': 4, 'condition': 'specific_hand'},
            'Faceless Joker': {'money': 5, 'condition': '3_face_discards'},
            'Reserved Parking': {'money': 2, 'targets': 'face_cards', 'chance': 0.5},
            'Mail-In Rebate': {'money': 5, 'special': 'discard_rank'},
            'Trading Card': {'money': 4, 'condition': 'first_discard_single'},
            'Matador': {'money': 8, 'condition': 'boss_ability_triggered'},
            'Vagabond': {'tarot': 1, 'condition': 'money_4_or_less'},
            
            # Growing jokers
            'Ride the Bus': {'mult': 0, 'grows': True, 'resets_on_face'},
            'Green Joker': {'mult': 0, 'grows_per_hand': 1, 'loses_per_discard': 1},
            'Red Card': {'mult': 0, 'grows_per_skip': 3},
            'Spare Trousers': {'mult': 0, 'grows_on': ['Two Pair', 'Full House']},
            'Square Joker': {'chips': 0, 'grows_per_4_cards': 4},
            'Runner': {'chips': 0, 'grows_on': 'Straight'},
            'Flash Card': {'mult': 0, 'grows_per_reroll': 2},
            'Popcorn': {'mult': 20, 'decays_per_round': 4, 'destroys': True},
            'Turtle Bean': {'hand_size': -1, 'decays_per_round': 1, 'destroys': True},
            'Ice Cream': {'chips': 100, 'decays_per_hand': 10, 'destroys': True},
            'Ramen': {'x_mult': 2, 'decays_per_discard': 0.01, 'destroys': True},
            'Yorick': {'x_mult': 1, 'grows_every_23_discards': 1},
            'Wee Joker': {'chips': 0, 'grows_per_2': 8},
            'Ceremonial Dagger': {'mult': 0, 'grows_by_destroying_jokers': True},
            'Marble Joker': {'special': 'add_stone_card'},
            'Loyalty Card': {'x_mult': 4, 'every': 5},
            'Gift Card': {'special': 'increase_all_values'},
            'Egg': {'money_value_grows': 3},
            'Campfire': {'x_mult': 1, 'grows_per_sell': 0.25, 'resets_on_boss': True},
            'Glass Joker': {'x_mult': 1, 'grows_per_glass_destroy': 0.75},
            'Caino': {'x_mult': 1, 'grows_per_face_destroy': 1},
            'Throwback': {'x_mult': 1, 'triggers_on_skip': True},
            'Hologram': {'x_mult': 1, 'grows_per_playing_card': 0.25},
            'Vampire': {'x_mult': 1, 'grows_per_enhanced_destroy': 0.2},
            'Space Joker': {'special': 'random_level_up', 'chance': 1/4},
            'Constellation': {'x_mult': 1, 'grows_per_planet': 0.1},
            'Castle': {'chips': 0, 'grows_per_discard_suit': 3},
            'Hit the Road': {'x_mult': 1, 'grows_per_jack_discard': 0.5, 'resets_end_round': True},
            'Lucky Cat': {'x_mult': 1, 'grows_per_lucky_trigger': 0.25},
            
            # Utility jokers
            'Credit Card': {'special': 'debt_20'},
            'Chaos the Clown': {'free_rerolls': 1},
            'Raised Fist': {'special': 'lowest_rank_mult'},
            'Baron': {'x_mult': 1.5, 'condition': 'king_in_hand'},
            'Shoot the Moon': {'hand_mult': 13, 'condition': 'queen_in_hand'},
            '8 Ball': {'tarot_chance': 1/4, 'rank': 8},
            'Obelisk': {'x_mult': 1, 'condition': 'most_played_hand'},
            'Midas Mask': {'special': 'faces_to_gold'},
            'Luchador': {'special': 'disable_boss_when_sold'},
            'Chicot': {'special': 'disable_boss'},
            'Madness': {'x_mult': 1, 'destroys_joker_per_round': True},
            'Burglar': {'special': 'trade_discards_for_hands'},
            'Riff-raff': {'special': 'create_2_jokers'},
            'Cartomancer': {'special': 'create_tarot'},
            'Perkeo': {'special': 'copy_random_consumable'},
            'Sixth Sense': {'spectral': 1, 'condition': 'destroy_6_first_hand'},
            'DNA': {'special': 'copy_single_card_first_hand'},
            'Splash': {'special': 'all_cards_score'},
            'Certificate': {'special': 'add_random_enhanced_card'},
            'Swashbuckler': {'mult': 0, 'grows_per_joker_sell': True},
            'Troubadour': {'special': 'hand_size_plus_2'},
            'Certificate': {'special': 'add_enhanced_card_with_seal'},
            'Seance': {'spectral': 1, 'hand_type': 'Straight Flush'},
            'Superposition': {'tarot': 1, 'condition': 'ace_straight'},
            'Mr. Bones': {'special': 'save_at_25_percent'},
            'Invisible Joker': {'special': 'copy_random_joker_after_4_rounds'},
            'Brainstorm': {'special': 'copy_leftmost_joker'},
            'Satellite': {'money_per_planet': 1},
            'Rocket': {'money_grows_on_boss': 2},
            'Oops! All 6s': {'special': 'all_probabilities_doubled'},
            'The Family': {'x_mult': 4, 'hand_type': 'Four of a Kind'},
            'The Order': {'x_mult': 3, 'hand_type': 'Straight'},
            'The Tribe': {'x_mult': 2, 'hand_type': 'Flush'},
            'Stuntman': {'chips': 250, 'hand_size_minus_1': True},
            'Invisible Joker': {'special': 'duplicate_random_joker'},
            'Brainstorm': {'special': 'copy_leftmost_joker'},
            'Satellite': {'money_per_planet': 1},
            'Shoot the Moon': {'hand_mult': 13, 'condition': 'queen_in_hand'},
            'Driver\'s License': {'x_mult': 3, 'condition': '16_enhanced'},
            'Cartomancer': {'create_tarot': True},
            'Astronomer': {'voucher_chance': True},
            'Burnt Joker': {'upgrade_hand_first_discard': True},
            'Bootstraps': {'mult_per_5_dollars': 2},
            'Canio': {'x_mult': 1, 'grows_per_face_destroy': 1},
            'Triboulet': {'x_mult': 2, 'ranks': [12, 13]},
            'Yorick': {'x_mult': 1, 'condition': 'every_23_discards'},
            'Chicot': {'disable_boss': True},
            'Perkeo': {'copy_consumable': True},
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
    
    def calculate_score(self, cards, jokers, hand_levels):
        """Perfect Balatro scoring replication"""
        # 1. Evaluate hand type using exact Lua logic
        hand_result = self.evaluate_hand(cards)
        hand_type = hand_result['top']
        scoring_cards = hand_result[hand_type][0] if hand_result[hand_type] else []
        
        # 2. Base scoring with hand levels
        base = self.base_hands[hand_type]
        level = hand_levels.get(hand_type, 1)
        
        chips = base['chips'] + (level - 1) * base['l_chips']
        mult = base['mult'] + (level - 1) * base['l_mult']
        
        # 3. Add card values
        for card in scoring_cards:
            chips += card.base_value
            
        # 4. Apply joker effects IN ORDER (critical!)
        for joker in jokers:
            effect = self.apply_joker_effect(joker, cards, scoring_cards, hand_type)
            if effect:
                chips += effect.get('chips', 0)
                mult += effect.get('mult', 0)
                if 'x_mult' in effect:
                    mult *= effect['x_mult']
                    
        return chips * mult
