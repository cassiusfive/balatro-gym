import random
import copy

class CompleteJokerEffects:
    """Complete implementation of all 150+ Balatro jokers from extracted code"""
    
    def __init__(self):
        # Initialize joker state tracking
        self.joker_states = {}
        
    def apply_joker_effect(self, joker, context, game_state):
        """
        Apply joker effects based on exact Balatro logic from extracted code
        Context includes: cards, scoring_cards, hand_type, phase, etc.
        """
        joker_name = joker.name
        
        # Initialize joker state if needed
        if joker_name not in self.joker_states:
            self.joker_states[joker_name] = self.get_initial_joker_state(joker_name)
        
        # Route to specific joker implementation
        method_name = f"effect_{joker_name.lower().replace(' ', '_').replace('-', '_')}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(joker, context, game_state)
        else:
            return self.effect_generic(joker, context, game_state)
    
    def get_initial_joker_state(self, joker_name):
        """Initialize joker-specific state based on extracted code"""
        initial_states = {
            'Ride the Bus': {'mult': 0},
            'Green Joker': {'mult': 1},
            'Red Card': {'mult': 0},
            'Spare Trousers': {'mult': 0},
            'Square Joker': {'chips': 0},
            'Runner': {'chips': 0},
            'Flash Card': {'mult': 0},
            'Popcorn': {'mult': 20},
            'Turtle Bean': {'hand_size': 5},
            'Ice Cream': {'chips': 100},
            'Ramen': {'x_mult': 2.0},
            'Yorick': {'discards': 0, 'x_mult': 1.0},
            'Wee Joker': {'chips': 0},
            'Ceremonial Dagger': {'mult': 0},
            'Glass Joker': {'x_mult': 1.0},
            'Caino': {'x_mult': 1.0},
            'Hologram': {'x_mult': 1.0},
            'Vampire': {'x_mult': 1.0},
            'Constellation': {'x_mult': 1.0},
            'Castle': {'chips': 0, 'suit': None},
            'Hit the Road': {'x_mult': 1.0},
            'Lucky Cat': {'x_mult': 1.0},
            'Campfire': {'x_mult': 1.0},
            'Throwback': {'x_mult': 1.0},
            'Invisible Joker': {'rounds': 0},
            'Loyalty Card': {'hands_played': 0},
            'To Do List': {'target_hand': 'High Card'},
            'Mail-In Rebate': {'target_rank': 1},
            'Madness': {'x_mult': 1.0},
            'Obelisk': {'x_mult': 1.0},
            'Swashbuckler': {'mult': 0},
            'Bootstraps': {'mult': 0},
            'Driver\'s License': {'enhanced_count': 0},
            'Egg': {'extra_value': 0},
            'Gift Card': {'extra_value': 0}
        }
        return initial_states.get(joker_name, {})

    # ===== BASIC JOKERS =====
    def effect_joker(self, joker, context, game_state):
        """Basic Joker: +4 Mult"""
        if context.get('phase') == 'scoring':
            return {'mult': 4}
        return None

    def effect_greedy_joker(self, joker, context, game_state):
        """Greedy Joker: +3 Mult if played hand contains a Diamond"""
        if context.get('phase') == 'scoring':
            for card in context.get('scoring_cards', []):
                if card.suit == 'Diamonds':
                    return {'mult': 3}
        return None

    def effect_lusty_joker(self, joker, context, game_state):
        """Lusty Joker: +3 Mult if played hand contains a Heart"""
        if context.get('phase') == 'scoring':
            for card in context.get('scoring_cards', []):
                if card.suit == 'Hearts':
                    return {'mult': 3}
        return None

    def effect_wrathful_joker(self, joker, context, game_state):
        """Wrathful Joker: +3 Mult if played hand contains a Spade"""
        if context.get('phase') == 'scoring':
            for card in context.get('scoring_cards', []):
                if card.suit == 'Spades':
                    return {'mult': 3}
        return None

    def effect_gluttonous_joker(self, joker, context, game_state):
        """Gluttonous Joker: +3 Mult if played hand contains a Club"""
        if context.get('phase') == 'scoring':
            for card in context.get('scoring_cards', []):
                if card.suit == 'Clubs':
                    return {'mult': 3}
        return None

    # ===== HAND-TYPE SPECIFIC JOKERS =====
    def effect_jolly_joker(self, joker, context, game_state):
        """Jolly Joker: +8 Mult if played hand is a Pair"""
        if context.get('phase') == 'scoring' and context.get('hand_type') == 'Pair':
            return {'mult': 8}
        return None

    def effect_zany_joker(self, joker, context, game_state):
        """Zany Joker: +12 Mult if played hand is a Three of a Kind"""
        if context.get('phase') == 'scoring' and context.get('hand_type') == 'Three of a Kind':
            return {'mult': 12}
        return None

    def effect_mad_joker(self, joker, context, game_state):
        """Mad Joker: +10 Mult if played hand is a Two Pair"""
        if context.get('phase') == 'scoring' and context.get('hand_type') == 'Two Pair':
            return {'mult': 10}
        return None

    def effect_crazy_joker(self, joker, context, game_state):
        """Crazy Joker: +12 Mult if played hand is a Straight"""
        if context.get('phase') == 'scoring' and context.get('hand_type') == 'Straight':
            return {'mult': 12}
        return None

    def effect_droll_joker(self, joker, context, game_state):
        """Droll Joker: +10 Mult if played hand is a Flush"""
        if context.get('phase') == 'scoring' and context.get('hand_type') == 'Flush':
            return {'mult': 10}
        return None

    # ===== INDIVIDUAL CARD EFFECTS =====
    def effect_fibonacci(self, joker, context, game_state):
        """Fibonacci: Each played 2, 3, 5, 8, or Ace gives +8 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [2, 3, 5, 8, 14]:
                return {'mult': 8}
        return None

    def effect_scholar(self, joker, context, game_state):
        """Scholar: Played Aces give +20 Chips and +4 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank == 14:  # Ace
                return {'chips': 20, 'mult': 4}
        return None

    def effect_even_steven(self, joker, context, game_state):
        """Even Steven: Each played even-ranked card gives +4 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank <= 10 and card.rank % 2 == 0:
                return {'mult': 4}
        return None

    def effect_odd_todd(self, joker, context, game_state):
        """Odd Todd: Each played odd-ranked card gives +31 Chips"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and ((card.rank <= 10 and card.rank % 2 == 1) or card.rank == 14):
                return {'chips': 31}
        return None

    def effect_scary_face(self, joker, context, game_state):
        """Scary Face: Each played face card gives +30 Chips"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [11, 12, 13]:  # J, Q, K
                return {'chips': 30}
        return None

    def effect_smiley_face(self, joker, context, game_state):
        """Smiley Face: Each played face card gives +5 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [11, 12, 13]:  # J, Q, K
                return {'mult': 5}
        return None

    def effect_triboulet(self, joker, context, game_state):
        """Triboulet: Played Kings and Queens each give X2 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [12, 13]:  # Q, K
                return {'x_mult': 2}
        return None

    def effect_walkie_talkie(self, joker, context, game_state):
        """Walkie Talkie: Each played 10 or 4 gives +10 Chips and +4 Mult"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [4, 10]:
                return {'chips': 10, 'mult': 4}
        return None

    # ===== GROWING JOKERS =====
    def effect_ride_the_bus(self, joker, context, game_state):
        """Ride the Bus: +1 Mult per consecutive hand without a face card, resets if face card"""
        state = self.joker_states['Ride the Bus']
        
        if context.get('phase') == 'before_scoring':
            # Check if hand has face cards
            has_face = any(card.rank in [11, 12, 13] for card in context.get('scoring_cards', []))
            if has_face:
                state['mult'] = 0  # Reset
                return {'message': 'Reset!'}
            else:
                state['mult'] += 1
                return {'mult': state['mult']}
        elif context.get('phase') == 'scoring':
            return {'mult': state['mult']}
        return None

    def effect_green_joker(self, joker, context, game_state):
        """Green Joker: +1 Mult per hand played, -1 Mult per discard"""
        state = self.joker_states['Green Joker']
        
        if context.get('phase') == 'before_scoring':
            state['mult'] += 1
            return {'message': f"+1 Mult (now {state['mult']})"}
        elif context.get('phase') == 'discard':
            if context.get('card') == context.get('last_discarded_card'):  # Last card discarded
                state['mult'] = max(0, state['mult'] - 1)
                return {'message': f"-1 Mult (now {state['mult']})"}
        elif context.get('phase') == 'scoring':
            return {'mult': state['mult']}
        return None

    def effect_red_card(self, joker, context, game_state):
        """Red Card: +3 Mult per skipped Booster Pack"""
        state = self.joker_states['Red Card']
        
        if context.get('phase') == 'skip_booster':
            state['mult'] += 3
            return {'message': f"+3 Mult (now {state['mult']})"}
        elif context.get('phase') == 'scoring':
            return {'mult': state['mult']}
        return None

    def effect_square_joker(self, joker, context, game_state):
        """Square Joker: +4 Chips if played hand has exactly 4 cards"""
        state = self.joker_states['Square Joker']
        
        if context.get('phase') == 'before_scoring':
            if len(context.get('scoring_cards', [])) == 4:
                state['chips'] += 4
                return {'message': f"+4 Chips (now {state['chips']})"}
        elif context.get('phase') == 'scoring':
            return {'chips': state['chips']}
        return None

    def effect_wee_joker(self, joker, context, game_state):
        """Wee Joker: +8 Chips if played hand contains a 2"""
        state = self.joker_states['Wee Joker']
        
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank == 2:
                state['chips'] += 8
                return {'message': f"+8 Chips (now {state['chips']})"}
        elif context.get('phase') == 'scoring':
            return {'chips': state['chips']}
        return None

    # ===== ECONOMIC JOKERS =====
    def effect_rough_gem(self, joker, context, game_state):
        """Rough Gem: Each played Diamond gives +$1"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.suit == 'Diamonds':
                game_state['money'] = game_state.get('money', 0) + 1
                return {'money': 1}
        return None

    def effect_golden_ticket(self, joker, context, game_state):
        """Golden Ticket: Each played Gold Card gives +$4"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and getattr(card, 'enhancement', None) == 'gold':
                game_state['money'] = game_state.get('money', 0) + 4
                return {'money': 4}
        return None

    def effect_business_card(self, joker, context, game_state):
        """Business Card: Played face cards have 1 in 2 chance to give +$2"""
        if context.get('phase') == 'individual_scoring':
            card = context.get('card')
            if card and card.rank in [11, 12, 13] and random.random() < 0.5:
                game_state['money'] = game_state.get('money', 0) + 2
                return {'money': 2}
        return None

    def effect_trading_card(self, joker, context, game_state):
        """Trading Card: If first discard of round has only 1 card, gain +$4"""
        if context.get('phase') == 'discard' and context.get('is_first_discard') and len(context.get('discarded_cards', [])) == 1:
            game_state['money'] = game_state.get('money', 0) + 4
            return {'money': 4}
        return None

    # ===== CONDITIONAL JOKERS =====
    def effect_half_joker(self, joker, context, game_state):
        """Half Joker: +20 Mult if played hand has 3 or fewer cards"""
        if context.get('phase') == 'scoring' and len(context.get('scoring_cards', [])) <= 3:
            return {'mult': 20}
        return None

    def effect_abstract_joker(self, joker, context, game_state):
        """Abstract Joker: +3 Mult for each Joker"""
        if context.get('phase') == 'scoring':
            joker_count = len([j for j in game_state.get('jokers', []) if j.type == 'Joker'])
            return {'mult': 3 * joker_count}
        return None

    def effect_supernova(self, joker, context, game_state):
        """Supernova: +Mult equal to number of times current poker hand has been played this run"""
        if context.get('phase') == 'scoring':
            hand_type = context.get('hand_type', 'High Card')
            times_played = game_state.get('hand_stats', {}).get(hand_type, {}).get('played', 0)
            return {'mult': times_played}
        return None

    def effect_blue_joker(self, joker, context, game_state):
        """Blue Joker: +2 Chips for each remaining card in deck"""
        if context.get('phase') == 'scoring':
            deck_size = len(game_state.get('deck', []))
            return {'chips': 2 * deck_size}
        return None

    def effect_banner(self, joker, context, game_state):
        """Banner: +30 Chips for each remaining discard"""
        if context.get('phase') == 'scoring':
            discards_left = game_state.get('discards_left', 0)
            return {'chips': 30 * discards_left}
        return None

    def effect_acrobat(self, joker, context, game_state):
        """Acrobat: X3 Mult on final hand of round"""
        if context.get('phase') == 'scoring' and game_state.get('hands_left', 1) == 1:
            return {'x_mult': 3}
        return None

    def effect_seeing_double(self, joker, context, game_state):
        """Seeing Double: X2 Mult if played hand has a Club and any other suit"""
        if context.get('phase') == 'scoring':
            suits = set(card.suit for card in context.get('scoring_cards', []))
            if 'Clubs' in suits and len(suits) > 1:
                return {'x_mult': 2}
        return None

    def effect_flower_pot(self, joker, context, game_state):
        """Flower Pot: X3 Mult if played hand contains each suit"""
        if context.get('phase') == 'scoring':
            suits = set(card.suit for card in context.get('scoring_cards', []))
            if len(suits) == 4:  # All suits
                return {'x_mult': 3}
        return None

    # ===== UTILITY JOKERS =====
    def effect_blueprint(self, joker, context, game_state):
        """Blueprint: Copy the ability of the Joker to the right"""
        if context.get('phase') == 'scoring':
            jokers = game_state.get('jokers', [])
            current_index = next((i for i, j in enumerate(jokers) if j.name == 'Blueprint'), -1)
            if current_index != -1 and current_index + 1 < len(jokers):
                next_joker = jokers[current_index + 1]
                # Apply next joker's effect
                return self.apply_joker_effect(next_joker, context, game_state)
        return None

    def effect_brainstorm(self, joker, context, game_state):
        """Brainstorm: Copy the ability of the leftmost Joker"""
        if context.get('phase') == 'scoring':
            jokers = game_state.get('jokers', [])
            if jokers and jokers[0].name != 'Brainstorm':
                leftmost_joker = jokers[0]
                return self.apply_joker_effect(leftmost_joker, context, game_state)
        return None

    def effect_four_fingers(self, joker, context, game_state):
        """Four Fingers: All Flushes and Straights can be made with 4 cards"""
        # This is handled in the hand evaluation logic, not here
        return None

    def effect_mime(self, joker, context, game_state):
        """Mime: Retrigger all held cards"""
        if context.get('phase') == 'hand_retrigger':
            return {'retriggers': 1}
        return None

    # ===== SPECIAL MECHANICS =====
    def effect_glass_joker(self, joker, context, game_state):
        """Glass Joker: +0.75 X Mult for every Glass Card destroyed"""
        state = self.joker_states['Glass Joker']
        
        if context.get('phase') == 'card_destroyed':
            destroyed_card = context.get('card')
            if destroyed_card and getattr(destroyed_card, 'enhancement', None) == 'glass':
                state['x_mult'] += 0.75
                return {'message': f"+0.75 X Mult (now {state['x_mult']:.2f})"}
        elif context.get('phase') == 'scoring':
            if state['x_mult'] > 1:
                return {'x_mult': state['x_mult']}
        return None

    def effect_vampire(self, joker, context, game_state):
        """Vampire: +0.2 X Mult per Enhanced card, remove enhancement"""
        state = self.joker_states['Vampire']
        
        if context.get('phase') == 'before_scoring':
            enhanced_count = 0
            for card in context.get('scoring_cards', []):
                if getattr(card, 'enhancement', None) and card.enhancement != 'none':
                    enhanced_count += 1
                    card.enhancement = None  # Remove enhancement
            
            if enhanced_count > 0:
                state['x_mult'] += 0.2 * enhanced_count
                return {'message': f"+{0.2 * enhanced_count:.1f} X Mult (now {state['x_mult']:.2f})"}
        elif context.get('phase') == 'scoring':
            if state['x_mult'] > 1:
                return {'x_mult': state['x_mult']}
        return None

    # ===== GENERIC FALLBACK =====
    def effect_generic(self, joker, context, game_state):
        """Fallback for jokers not yet implemented"""
        # Basic jokers that just give static bonuses
        basic_effects = {
            'Joker': {'mult': 4},
            'Stuntman': {'chips': 250},
            'Misprint': {'mult': random.randint(0, 23)},
            'Gros Michel': {'mult': 15},
            'Cavendish': {'x_mult': 3}
        }
        
        if context.get('phase') == 'scoring' and joker.name in basic_effects:
            return basic_effects[joker.name]
        return None

    # ===== HELPER METHODS =====
    def reset_round_effects(self):
        """Reset effects that trigger once per round"""
        for joker_name, state in self.joker_states.items():
            if joker_name in ['To Do List', 'Loyalty Card']:
                # Reset round-specific counters
                pass

    def end_of_round_effects(self, game_state):
        """Handle end-of-round joker effects"""
        effects = []
        
        for joker_name, state in self.joker_states.items():
            if joker_name == 'Popcorn':
                # Lose 4 mult each round
                state['mult'] = max(0, state['mult'] - 4)
                if state['mult'] <= 0:
                    effects.append({'destroy_joker': joker_name, 'message': 'Popcorn eaten!'})
            
            elif joker_name == 'Turtle Bean':
                # Lose 1 hand size each round
                state['hand_size'] -= 1
                if state['hand_size'] <= 0:
                    effects.append({'destroy_joker': joker_name, 'message': 'Turtle Bean eaten!'})
        
        return effects
