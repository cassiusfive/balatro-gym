"""Pack opening phase handler for Balatro RL environment.

This module handles the pack opening phase where players select
cards from booster packs.
"""

from typing import Tuple, Dict, List, Optional

from balatro_gym.envs.state import UnifiedGameState
from balatro_gym.constants import Action, Phase
from balatro_gym.cards import Card, Enhancement, Edition, Seal


class PackOpenHandler:
    """Handles pack opening phase."""
    
    def __init__(self, state: UnifiedGameState, shop_handler):
        """Initialize the pack open handler.
        
        Args:
            state: Game state
            shop_handler: Reference to shop handler for returning to shop
        """
        self.state = state
        self.shop_handler = shop_handler
        self.pack_contents: List[Dict] = []
        self.pack_type: str = ""
        self.cards_to_select: int = 1
        self.selected_indexes: List[int] = []
    
    def step(self, action: int) -> Tuple[float, bool, Dict]:
        """Process an action during pack open phase.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (reward, terminated, info)
        """
        if Action.PACK_SELECT_BASE <= action < Action.PACK_SELECT_BASE + Action.PACK_SELECT_COUNT:
            return self._handle_select_card(action)
        elif action == Action.PACK_SKIP:
            return self._handle_skip_pack()
        else:
            return -1.0, False, {'error': 'Invalid pack open action'}
    
    def open_pack(self, pack_type: str, pack_contents: List[Dict]) -> Dict:
        """Initialize pack opening with contents.
        
        Args:
            pack_type: Type of pack being opened
            pack_contents: List of card/item dictionaries in the pack
            
        Returns:
            Info dictionary about the pack
        """
        self.pack_type = pack_type
        self.pack_contents = pack_contents
        self.selected_indexes = []
        
        # Determine how many cards can be selected
        self.cards_to_select = self._get_cards_to_select(pack_type)
        
        # Transition to pack open phase
        self.state.phase = Phase.PACK_OPEN
        
        return {
            'pack_type': pack_type,
            'pack_size': len(pack_contents),
            'cards_to_select': self.cards_to_select,
            'pack_contents': self._format_pack_contents()
        }
    
    def _handle_select_card(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle selecting a card from the pack."""
        card_idx = action - Action.PACK_SELECT_BASE
        
        if card_idx >= len(self.pack_contents):
            return -1.0, False, {'error': 'Invalid card index'}
        
        if card_idx in self.selected_indexes:
            return -1.0, False, {'error': 'Card already selected'}
        
        if len(self.selected_indexes) >= self.cards_to_select:
            return -1.0, False, {'error': 'Already selected maximum cards'}
        
        # Select the card
        self.selected_indexes.append(card_idx)
        selected_item = self.pack_contents[card_idx]
        
        # Apply the selected item
        reward, apply_info = self._apply_pack_item(selected_item)
        
        info = {
            'action': 'selected_card',
            'card_index': card_idx,
            'cards_selected': len(self.selected_indexes),
            'cards_remaining': self.cards_to_select - len(self.selected_indexes)
        }
        info.update(apply_info)
        
        # Check if pack selection is complete
        if len(self.selected_indexes) >= self.cards_to_select:
            return self._complete_pack_opening(reward, info)
        
        return reward, False, info
    
    def _handle_skip_pack(self) -> Tuple[float, bool, Dict]:
        """Handle skipping the remaining pack selections."""
        # Small penalty for not using full pack value
        cards_skipped = self.cards_to_select - len(self.selected_indexes)
        reward = -1.0 * cards_skipped
        
        info = {
            'action': 'skipped_pack',
            'cards_skipped': cards_skipped
        }
        
        return self._complete_pack_opening(reward, info)
    
    def _complete_pack_opening(self, base_reward: float, info: Dict) -> Tuple[float, bool, Dict]:
        """Complete pack opening and return to shop."""
        # Clear pack state
        self.pack_contents = []
        self.pack_type = ""
        self.selected_indexes = []
        
        # Return to shop phase
        self.state.phase = Phase.SHOP
        
        # Regenerate shop display
        if self.shop_handler.shop:
            self.state.shop_inventory = self.shop_handler.shop.inventory.copy()
        
        info['transition_to'] = 'shop'
        
        return base_reward, False, info
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _get_cards_to_select(self, pack_type: str) -> int:
        """Get number of cards that can be selected from pack type."""
        pack_selections = {
            'Arcana Pack': 1,      # Choose 1 of 5 Tarot cards
            'Celestial Pack': 1,   # Choose 1 of 5 Planet cards
            'Spectral Pack': 1,    # Choose 1 of 5 Spectral cards
            'Standard Pack': 1,    # Choose 1 of 5 playing cards
            'Buffoon Pack': 1,     # Choose 1 of 5 Jokers
            'Mega Arcana Pack': 2, # Choose 2 of 5 Tarot cards
            'Mega Celestial Pack': 2,  # Choose 2 of 5 Planet cards
            'Mega Spectral Pack': 2,   # Choose 2 of 5 Spectral cards
            'Mega Standard Pack': 2,   # Choose 2 of 5 playing cards
            'Mega Buffoon Pack': 2,    # Choose 2 of 5 Jokers
        }
        
        return pack_selections.get(pack_type, 1)
    
    def _format_pack_contents(self) -> List[str]:
        """Format pack contents for display."""
        formatted = []
        
        for item in self.pack_contents:
            if 'card' in item:
                # Playing card
                card = item['card']
                desc = f"{card.rank.name} of {card.suit.name}"
                if 'enhancement' in item and item['enhancement'] != Enhancement.NONE:
                    desc += f" ({item['enhancement'].name})"
                if 'edition' in item and item['edition'] != Edition.NONE:
                    desc += f" [{item['edition'].name}]"
                if 'seal' in item and item['seal'] != Seal.NONE:
                    desc += f" <{item['seal'].name}>"
                formatted.append(desc)
            
            elif 'consumable' in item:
                # Tarot/Planet/Spectral card
                formatted.append(item['consumable'])
            
            elif 'joker' in item:
                # Joker
                formatted.append(f"Joker: {item['joker'].name}")
            
            else:
                formatted.append("Unknown item")
        
        return formatted
    
    def _apply_pack_item(self, item: Dict) -> Tuple[float, Dict]:
        """Apply the selected pack item to game state."""
        info = {}
        reward = 0.0
        
        if 'card' in item:
            # Add playing card to deck
            card = item['card']
            card_idx = len(self.state.deck)
            self.state.deck.append(card)
            
            # Apply any enhancements/editions/seals
            if any(key in item for key in ['enhancement', 'edition', 'seal']):
                card_state = self.state.get_card_state(card_idx)
                card_state.enhancement = item.get('enhancement', Enhancement.NONE)
                card_state.edition = item.get('edition', Edition.NONE)
                card_state.seal = item.get('seal', Seal.NONE)
            
            info['card_added'] = f"{card.rank.name} of {card.suit.name}"
            reward = 3.0  # Base value for adding a card
            
            # Bonus for enhanced cards
            if item.get('enhancement', Enhancement.NONE) != Enhancement.NONE:
                reward += 2.0
            if item.get('edition', Edition.NONE) != Edition.NONE:
                reward += 3.0
            if item.get('seal', Seal.NONE) != Seal.NONE:
                reward += 2.0
        
        elif 'consumable' in item:
            # Add consumable to inventory
            if len(self.state.consumables) < self.state.consumable_slots:
                self.state.consumables.append(item['consumable'])
                info['consumable_added'] = item['consumable']
                
                # Value based on consumable type
                if 'Planet' in item['consumable'] or item['consumable'].startswith(('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto')):
                    reward = 8.0  # Planets are very valuable
                elif 'The' in item['consumable']:  # Tarot cards
                    reward = 5.0
                else:  # Spectral cards
                    reward = 10.0  # Spectral cards are rare and powerful
            else:
                info['error'] = 'No consumable slots available'
                reward = -1.0
        
        elif 'joker' in item:
            # Add joker to collection
            if self.state.add_joker(item['joker']):
                info['joker_added'] = item['joker'].name
                reward = 15.0  # Jokers are very valuable
                
                # Extra reward for rare jokers
                if item['joker'].rarity == 'Legendary':
                    reward += 10.0
                elif item['joker'].rarity == 'Rare':
                    reward += 5.0
            else:
                info['error'] = 'No joker slots available'
                reward = -1.0
        
        return reward, info
