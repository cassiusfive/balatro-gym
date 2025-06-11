"""Shop phase handler for Balatro RL environment.

This module handles all actions during the SHOP phase including:
- Buying items (jokers, cards, vouchers, packs)
- Rerolling shop
- Selling jokers
- Ending shopping
"""

from typing import Tuple, Dict, Optional
import numpy as np

from balatro_gym.envs.state import UnifiedGameState
from balatro_gym.envs.rng import DeterministicRNG
from balatro_gym.constants import Action
from balatro_gym.shop import Shop, ShopAction, PlayerState, ItemType
from balatro_gym.jokers import JOKER_LIBRARY


class ShopPhaseHandler:
    """Handles all actions during the SHOP phase."""
    
    def __init__(self, state: UnifiedGameState, rng: DeterministicRNG):
        """Initialize the shop phase handler.
        
        Args:
            state: Game state
            rng: RNG system
        """
        self.state = state
        self.rng = rng
        self.shop: Optional[Shop] = None
    
    def step(self, action: int) -> Tuple[float, bool, Dict]:
        """Process an action during shop phase.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (reward, terminated, info)
        """
        if self.shop is None:
            self.generate_shop()
        
        if action == Action.SHOP_END:
            return self._handle_end_shop()
        elif action == Action.SHOP_REROLL:
            return self._handle_reroll()
        elif Action.SHOP_BUY_BASE <= action < Action.SHOP_BUY_BASE + Action.SHOP_BUY_COUNT:
            return self._handle_buy_item(action)
        elif Action.SELL_JOKER_BASE <= action < Action.SELL_JOKER_BASE + Action.SELL_JOKER_COUNT:
            return self._handle_sell_joker(action)
        else:
            return -1.0, False, {'error': 'Invalid shop action'}
    
    def generate_shop(self):
        """Generate a new shop with items."""
        # Create or update player state
        player_state = self._create_player_state()
        
        # Generate shop with RNG
        shop_seed = self.rng.get_int('shop_generation', 0, 2**31 - 1)
        self.shop = Shop(self.state.ante, player_state, seed=shop_seed)
        
        # Sync inventory
        self.state.shop_inventory = self.shop.inventory.copy()
        self.state.shop_reroll_cost = int(self.shop.reroll_cost * self.shop._cost_mult())
        self.state.shop_visits += 1
    
    def _handle_end_shop(self) -> Tuple[float, bool, Dict]:
        """Handle ending the shopping phase."""
        # Transition to play phase
        self.state.phase = Phase.PLAY
        
        # Draw initial hand for next round
        # This should be handled by the main environment
        
        return 0.0, False, {'action': 'shop_ended'}
    
    def _handle_reroll(self) -> Tuple[float, bool, Dict]:
        """Handle rerolling the shop."""
        if self.state.money < self.state.shop_reroll_cost:
            return -1.0, False, {'error': 'Cannot afford reroll'}
        
        # Execute reroll
        self._sync_player_state()
        reward, _, shop_info = self.shop.step(ShopAction.REROLL)
        
        # Update state
        self.state.money = self.shop.player.chips
        self.state.shop_inventory = self.shop.inventory.copy()
        self.state.rerolls_used += 1
        
        # Increase reroll cost
        self.state.shop_reroll_cost = int(self.shop.reroll_cost * self.shop._cost_mult())
        
        info = {
            'action': 'rerolled',
            'new_reroll_cost': self.state.shop_reroll_cost,
            'money_spent': self.state.shop_reroll_cost
        }
        info.update(shop_info)
        
        return reward, False, info
    
    def _handle_buy_item(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle buying an item from the shop."""
        item_idx = action - Action.SHOP_BUY_BASE
        
        if not (0 <= item_idx < len(self.shop.inventory)):
            return -1.0, False, {'error': 'Invalid item index'}
        
        item = self.shop.inventory[item_idx]
        
        # Check affordability
        if self.state.money < item.cost:
            return -1.0, False, {'error': f'Cannot afford {item.name} (costs ${item.cost})'}
        
        # Map to shop action
        if item.item_type == ItemType.PACK:
            shop_action = ShopAction.BUY_PACK_BASE + item_idx
        elif item.item_type == ItemType.JOKER:
            # Check joker slot availability
            if len(self.state.jokers) >= self.state.joker_slots:
                return -1.0, False, {'error': 'No joker slots available'}
            shop_action = ShopAction.BUY_JOKER_BASE + item_idx
        elif item.item_type == ItemType.CARD:
            shop_action = ShopAction.BUY_CARD_BASE + item_idx
        elif item.item_type == ItemType.VOUCHER:
            shop_action = ShopAction.BUY_VOUCHER_BASE + item_idx
        else:
            return -1.0, False, {'error': f'Unknown item type: {item.item_type}'}
        
        # Execute purchase
        self._sync_player_state()
        reward, _, shop_info = self.shop.step(shop_action)
        
        # Handle errors from shop
        if 'error' in shop_info:
            return -1.0, False, shop_info
        
        # Update state based on purchase type
        info = self._process_purchase(item, shop_info)
        
        # Update money and inventory
        self.state.money = self.shop.player.chips
        self.state.shop_inventory = self.shop.inventory.copy()
        
        # Calculate reward based on item type
        if item.item_type == ItemType.PACK:
            reward = 5.0  # Packs open new opportunities
        elif item.item_type == ItemType.JOKER:
            reward = 15.0  # Jokers are high value
        elif item.item_type == ItemType.CARD:
            reward = 3.0  # Cards are moderate value
        elif item.item_type == ItemType.VOUCHER:
            reward = 10.0  # Vouchers provide permanent benefits
        
        return reward, False, info
    
    def _handle_sell_joker(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle selling a joker."""
        joker_idx = action - Action.SELL_JOKER_BASE
        
        if not (0 <= joker_idx < len(self.state.jokers)):
            return -1.0, False, {'error': 'Invalid joker index'}
        
        # Check if joker is eternal
        if joker_idx in self.state.eternal_jokers:
            return -1.0, False, {'error': 'Cannot sell eternal jokers'}
        
        # Remove and sell joker
        sold_joker = self.state.remove_joker(joker_idx)
        if sold_joker is None:
            return -1.0, False, {'error': 'Failed to remove joker'}
        
        # Calculate sell value
        sell_value = self._calculate_sell_value(sold_joker)
        self.state.money += sell_value
        self.state.jokers_sold += 1
        
        # Sync with shop player state
        self._sync_player_state()
        
        # Calculate reward (small positive to allow strategic selling)
        reward = sell_value / 10.0
        
        # Special joker sale effects
        sale_effects = self._apply_joker_sale_effects(sold_joker)
        
        info = {
            'action': 'sold_joker',
            'joker_sold': sold_joker.name,
            'money_gained': sell_value,
            'jokers_remaining': len(self.state.jokers)
        }
        info.update(sale_effects)
        
        return reward, False, info
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _create_player_state(self) -> PlayerState:
        """Create player state from unified state."""
        player = PlayerState(chips=self.state.money)
        player.jokers = [j.id for j in self.state.jokers]
        player.vouchers = self.state.vouchers.copy()
        return player
    
    def _sync_player_state(self):
        """Sync player state with unified state."""
        if self.shop and self.shop.player:
            self.shop.player.chips = self.state.money
            self.shop.player.jokers = [j.id for j in self.state.jokers]
            self.shop.player.vouchers = self.state.vouchers.copy()
    
    def _sync_jokers_from_player(self):
        """Sync jokers from player state back to unified state."""
        if self.shop and self.shop.player:
            # Get new jokers that were added
            current_joker_ids = {j.id for j in self.state.jokers}
            for joker_id in self.shop.player.jokers:
                if joker_id not in current_joker_ids:
                    # Find joker info and add it
                    for joker_info in JOKER_LIBRARY:
                        if joker_info.id == joker_id:
                            self.state.add_joker(joker_info)
                            break
    
    def _process_purchase(self, item, shop_info: Dict) -> Dict:
        """Process purchase results based on item type."""
        info = {
            'action': 'bought_item',
            'item_name': item.name,
            'item_type': item.item_type.name,
            'cost': item.cost
        }
        
        if item.item_type == ItemType.PACK:
            # Pack will transition to pack opening phase
            if 'new_cards' in shop_info:
                info['pack_contents'] = shop_info['new_cards']
                info['transition_to'] = 'pack_open'
        
        elif item.item_type == ItemType.JOKER:
            # Sync joker from player state
            self._sync_jokers_from_player()
            if self.state.jokers:
                info['joker_acquired'] = self.state.jokers[-1].name
        
        elif item.item_type == ItemType.CARD:
            # Card was added to deck
            info['card_added'] = True
            if 'card_added' in shop_info:
                info['card_details'] = shop_info['card_added']
        
        elif item.item_type == ItemType.VOUCHER:
            # Sync vouchers
            self.state.vouchers = self.shop.player.vouchers.copy()
            if self.state.vouchers:
                info['voucher_acquired'] = self.state.vouchers[-1]
                info['voucher_effect'] = self._get_voucher_effect(self.state.vouchers[-1])
        
        return info
    
    def _calculate_sell_value(self, joker) -> int:
        """Calculate the sell value of a joker."""
        base_value = max(3, joker.base_cost // 2)
        
        # Some jokers might have special sell values
        special_sell_values = {
            'Egg': 5,  # Egg gains value over time
            'Gift Card': 0,  # Gift cards can't be sold
        }
        
        if joker.name in special_sell_values:
            return special_sell_values[joker.name]
        
        return base_value
    
    def _apply_joker_sale_effects(self, joker) -> Dict:
        """Apply any special effects from selling specific jokers."""
        effects = {}
        
        # Some jokers have effects when sold
        if joker.name == 'Luchador':
            # Luchador disables itself when sold
            effects['luchador_effect'] = 'Boss blind disabled this round'
        elif joker.name == 'Swashbuckler':
            # Swashbuckler gives extra sell value based on jokers sold
            bonus = self.state.jokers_sold
            self.state.money += bonus
            effects['swashbuckler_bonus'] = bonus
        
        return effects
    
    def _get_voucher_effect(self, voucher_name: str) -> str:
        """Get description of voucher effect."""
        voucher_effects = {
            'Overstock': '+1 card slot in shop',
            'Clearance Sale': 'All items in shop are 25% off',
            'Hone': 'Foil, Holographic, and Polychrome cards appear 2X more often',
            'Reroll Surplus': 'Rerolls cost $2 less',
            'Crystal Ball': '+1 consumable slot',
            'Telescope': 'Celestial Packs always contain your most used poker hand\'s Planet card',
            'Grabber': '+1 hand per round',
            'Dusk': 'Tarot and Planet cards appear 2X more often in the shop',
            'Retcon': 'Rerolls cost $2 less (again)',
            'Paint Brush': '+1 hand size',
            'Overstock Plus': '+1 card slot in shop (again)',
            'Liquidation': 'All items in shop are 50% off',
            'Wasteful': 'Permanently gain +1 discard every round',
            'Tarot Merchant': 'Tarot cards appear 2X more often in the shop',
            'Planet Merchant': 'Planet cards appear 2X more often in the shop',
            'Seed Money': 'Gain $1 interest for every $5 you have at the end of the round',
        }
        
        return voucher_effects.get(voucher_name, 'Unknown voucher effect')


from balatro_gym.constants import Phase  # Add this import at the top
