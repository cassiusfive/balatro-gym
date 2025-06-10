"""Integration code to add Tarot/Spectral cards to the Balatro environment"""

# Add to your BalatroCompleteEnv class:

def _step_play_consumable(self, action: int):
    """Extended consumable handling with Tarot/Spectral effects"""
    consumable_idx = action - 10
    if consumable_idx >= len(self.consumables):
        return self._get_observation(), -1.0, False, False, {'error': 'Invalid consumable'}
    
    consumable_name = self.consumables[consumable_idx]
    
    # Get target cards if needed
    target_cards = []
    if len(self.selected_cards) > 0:
        # Convert selected cards to Card objects
        for idx in self.selected_cards:
            if idx < len(self.game.hand_indexes):
                card_idx = self.game.hand_indexes[idx]
                deck_card = self.game.deck[card_idx]
                # Create enhanced card object
                card = Card(
                    rank=Rank(deck_card.rank.value + 2),  # Convert from 0-12 to 2-14
                    suit=Suit(deck_card.suit.value),
                    enhancement=getattr(deck_card, 'enhancement', Enhancement.NONE),
                    edition=getattr(deck_card, 'edition', Edition.NONE),
                    seal=getattr(deck_card, 'seal', Seal.NONE)
                )
                target_cards.append(card)
    
    # Apply consumable effect
    result = self.consumable_manager.use_consumable(
        consumable_name,
        self.game_state,
        target_cards
    )
    
    # Handle results
    reward = 0.0
    info = {'consumable_used': consumable_name}
    
    if result['success']:
        # Remove consumable
        self.consumables.pop(consumable_idx)
        
        # Apply effects
        if result.get('money_gained', 0) > 0:
            self.player.chips += result['money_gained']
            reward += result['money_gained'] / 10.0
            
        if result.get('planet_used'):
            # Apply planet to score engine
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
                'Ceres': HandType.FLUSH_HOUSE,
                'Eris': HandType.FLUSH_FIVE
            }
            if result['planet_used'] in planet_map:
                self.engine.apply_planet(planet_map[result['planet_used']])
                reward += 10.0
        
        if result.get('cards_affected'):
            # Update actual deck cards with enhancements
            reward += len(result['cards_affected']) * 2.0
            
        if result.get('cards_created'):
            # Add new cards to deck
            for created_card in result['cards_created']:
                # Convert back to game card format
                new_card = self._create_game_card(created_card)
                self.game.deck.append(new_card)
            reward += len(result['cards_created']) * 3.0
            
        if result.get('cards_destroyed'):
            # Remove destroyed cards
            reward += len(result['cards_destroyed']) * 1.0  # Deck thinning bonus
            
        if result.get('jokers_created'):
            # Add created jokers
            for joker in result['jokers_created']:
                if len(self.jokers) < self.joker_slots:
                    self.jokers.append(joker)
            reward += len(result['jokers_created']) * 15.0
            
        if result.get('items_created'):
            # Add created consumables
            for item in result['items_created']:
                if len(self.consumables) < self.consumable_slots:
                    self.consumables.append(item)
            reward += len(result['items_created']) * 5.0
            
        if result.get('hand_size_change'):
            # Apply hand size changes
            self.game.hand_size += result['hand_size_change']
            
        info['result'] = result['message']
    else:
        reward = -1.0
        info['error'] = result.get('message', 'Failed to use consumable')
    
    # Clear selection after using consumable
    self.selected_cards = []
    self._update_game_state()
    
    return self._get_observation(), reward, False, False, info

def _create_game_card(self, card: Card):
    """Convert Card object back to game card format"""
    game_card = type('GameCard', (), {
        'rank': type('Rank', (), {'value': card.rank.value - 2})(),  # Convert back to 0-12
        'suit': type('Suit', (), {'value': card.suit.value})(),
        'played': False,
        'chip_value': lambda: 11 if card.rank == Rank.ACE else min(card.rank.value, 10),
        'encode': lambda: card.encode(),
        'enhancement': card.enhancement,
        'edition': card.edition,
        'seal': card.seal
    })
    return game_card

# Add to __init__ method:
self.consumable_manager = ConsumableManager()

# Update _get_action_mask to handle card selection for consumables:
def _get_action_mask_updated(self):
    """Updated action mask with consumable targeting"""
    mask = self._get_action_mask()  # Get base mask
    
    if self.phase == Phase.PLAY:
        # Check if any consumable needs card selection
        for i, consumable in enumerate(self.consumables):
            if i < 5:  # Max 5 consumables
                # Tarots that need card selection
                needs_selection = [
                    'The Magician', 'The Empress', 'The Hierophant',
                    'The Lovers', 'The Chariot', 'Strength', 'Justice',
                    'The Hanged Man', 'Death', 'The Devil', 'The Tower',
                    'The Star', 'The Moon', 'The Sun', 'The World'
                ]
                
                # Spectrals that need card selection
                needs_selection.extend([
                    'Familiar', 'Grim', 'Incantation', 'Talisman',
                    'Aura', 'Deja Vu', 'Trance', 'Medium', 'Cryptid'
                ])
                
                if consumable in needs_selection and len(self.selected_cards) == 0:
                    # Disable using this consumable until cards are selected
                    mask[10 + i] = 0
    
    return mask

# Add consumable names to observations:
def _get_consumable_ids(self):
    """Convert consumable names to IDs for observation"""
    consumable_id_map = {
        # Tarots
        'The Fool': 1, 'The Magician': 2, 'The High Priestess': 3,
        'The Empress': 4, 'The Emperor': 5, 'The Hierophant': 6,
        'The Lovers': 7, 'The Chariot': 8, 'Strength': 9,
        'The Hermit': 10, 'Wheel of Fortune': 11, 'Justice': 12,
        'The Hanged Man': 13, 'Death': 14, 'Temperance': 15,
        'The Devil': 16, 'The Tower': 17, 'The Star': 18,
        'The Moon': 19, 'The Sun': 20, 'Judgement': 21,
        'The World': 22,
        
        # Planets
        'Mercury': 30, 'Venus': 31, 'Earth': 32, 'Mars': 33,
        'Jupiter': 34, 'Saturn': 35, 'Uranus': 36, 'Neptune': 37,
        'Pluto': 38, 'Planet X': 39, 'Ceres': 40, 'Eris': 41,
        
        # Spectrals
        'Familiar': 50, 'Grim': 51, 'Incantation': 52, 'Talisman': 53,
        'Aura': 54, 'Wraith': 55, 'Sigil': 56, 'Ouija': 57,
        'Ectoplasm': 58, 'Immolate': 59, 'Ankh': 60, 'Deja Vu': 61,
        'Hex': 62, 'Trance': 63, 'Medium': 64, 'Cryptid': 65,
        'The Soul': 66, 'Black Hole': 67
    }
    
    ids = []
    for consumable in self.consumables:
        ids.append(consumable_id_map.get(consumable, 0))
    
    return ids + [0] * (5 - len(ids))

# Example usage in the environment:
"""
# When player selects cards and uses The Magician:
env.selected_cards = [0, 1]  # Select first two cards
obs, reward, done, truncated, info = env.step(10)  # Use first consumable

# The Magician will enhance those two cards to Lucky
# Reward will be based on the enhancement value
"""
