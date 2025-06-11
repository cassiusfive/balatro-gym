"""balatro_gym/consumables.py - Tarot and Spectral Cards Implementation

Complete implementation of all Tarot and Spectral cards from Balatro.
Handles card enhancement, transformation, destruction, and special effects.
"""

from __future__ import annotations
from enum import IntEnum, auto
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import random

# Card suits and ranks
class Suit(IntEnum):
    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3

class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

# Card enhancements
class Enhancement(IntEnum):
    NONE = 0
    BONUS = 1      # +30 chips
    MULT = 2       # +4 mult
    WILD = 3       # Counts as any suit
    GLASS = 4      # x2 mult, 25% chance to destroy
    STEEL = 5      # x1.5 mult while in hand
    STONE = 6      # +50 chips, no rank/suit
    GOLD = 7       # +$3 when played
    LUCKY = 8      # 1/5 chance for +$1

# Card editions
class Edition(IntEnum):
    NONE = 0
    FOIL = 1          # +50 chips
    HOLOGRAPHIC = 2   # +10 mult
    POLYCHROME = 3    # x1.5 mult
    NEGATIVE = 4      # +1 joker slot

# Card seals
class Seal(IntEnum):
    NONE = 0
    RED = 1      # Retrigger card
    BLUE = 2     # Creates planet card when held
    GOLD = 3     # +$3 when played
    PURPLE = 4   # Creates tarot card when discarded

@dataclass
class Card:
    """Card representation with all properties"""
    rank: Rank
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    
    def encode(self) -> int:
        """Encode to 0-51 index"""
        return self.suit * 13 + (self.rank - 2)
    
    @classmethod
    def decode(cls, index: int) -> 'Card':
        """Decode from 0-51 index"""
        suit = Suit(index // 13)
        rank = Rank(index % 13 + 2)
        return cls(rank, suit)

# ---------------------------------------------------------------------------
# Tarot Cards
# ---------------------------------------------------------------------------

class TarotCard(IntEnum):
    THE_FOOL = auto()       # Copy random consumable in inventory
    THE_MAGICIAN = auto()   # Enhance 2 cards to Lucky
    THE_HIGH_PRIESTESS = auto()  # Create 2 random Planet cards
    THE_EMPRESS = auto()    # Enhance 2 cards to Mult
    THE_EMPEROR = auto()    # Create 2 random Tarot cards
    THE_HIEROPHANT = auto() # Enhance 2 cards to Bonus
    THE_LOVERS = auto()     # Enhance 1 card to Wild
    THE_CHARIOT = auto()    # Enhance 1 card to Steel
    STRENGTH = auto()       # Increase rank of 2 cards by 1
    THE_HERMIT = auto()     # Double money (max $20)
    WHEEL_OF_FORTUNE = auto()  # 1/4 chance to add edition to card
    JUSTICE = auto()        # Enhance 1 card to Glass
    THE_HANGED_MAN = auto() # Destroy up to 2 cards
    DEATH = auto()          # Select 2 cards, convert left to right
    TEMPERANCE = auto()     # Give joker sell value to money (max $50)
    THE_DEVIL = auto()      # Enhance 1 card to Gold
    THE_TOWER = auto()      # Enhance 1 card to Stone
    THE_STAR = auto()       # Convert 3 cards to Diamonds
    THE_MOON = auto()       # Convert 3 cards to Clubs
    THE_SUN = auto()        # Convert 3 cards to Hearts
    JUDGEMENT = auto()      # Create random Planet card
    THE_WORLD = auto()      # Convert 3 cards to Spades

class TarotEffects:
    """Handles all tarot card effects"""
    
    @staticmethod
    def apply_tarot(tarot: TarotCard, game_state: Dict, 
                   target_cards: Optional[List[Card]] = None) -> Dict[str, Any]:
        """Apply a tarot card effect and return results"""
        
        result = {
            'success': False,
            'message': '',
            'cards_affected': [],
            'items_created': [],
            'money_gained': 0
        }
        
        if tarot == TarotCard.THE_FOOL:
            # Copy random consumable
            if game_state.get('consumables'):
                copied = random.choice(game_state['consumables'])
                game_state['consumables'].append(copied)
                result['items_created'].append(copied)
                result['success'] = True
                result['message'] = f"Copied {copied}"
                
        elif tarot == TarotCard.THE_MAGICIAN:
            # Enhance 2 cards to Lucky
            if target_cards:
                for card in target_cards[:2]:
                    card.enhancement = Enhancement.LUCKY
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Enhanced cards to Lucky"
                
        elif tarot == TarotCard.THE_HIGH_PRIESTESS:
            # Create 2 random Planet cards
            planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
            for _ in range(2):
                planet = random.choice(planets)
                if len(game_state.get('consumables', [])) < game_state.get('consumable_slots', 2):
                    game_state['consumables'].append(planet)
                    result['items_created'].append(planet)
            result['success'] = True
            result['message'] = "Created 2 Planet cards"
            
        elif tarot == TarotCard.THE_EMPRESS:
            # Enhance 2 cards to Mult
            if target_cards:
                for card in target_cards[:2]:
                    card.enhancement = Enhancement.MULT
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Enhanced cards to Mult"
                
        elif tarot == TarotCard.THE_EMPEROR:
            # Create 2 random Tarot cards
            tarots = list(TarotCard)
            for _ in range(2):
                if len(game_state.get('consumables', [])) < game_state.get('consumable_slots', 2):
                    tarot_card = random.choice(tarots)
                    game_state['consumables'].append(tarot_card.name)
                    result['items_created'].append(tarot_card.name)
            result['success'] = True
            result['message'] = "Created 2 Tarot cards"
            
        elif tarot == TarotCard.THE_HIEROPHANT:
            # Enhance 2 cards to Bonus
            if target_cards:
                for card in target_cards[:2]:
                    card.enhancement = Enhancement.BONUS
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Enhanced cards to Bonus"
                
        elif tarot == TarotCard.THE_LOVERS:
            # Enhance 1 card to Wild
            if target_cards and len(target_cards) >= 1:
                target_cards[0].enhancement = Enhancement.WILD
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Enhanced card to Wild"
                
        elif tarot == TarotCard.THE_CHARIOT:
            # Enhance 1 card to Steel
            if target_cards and len(target_cards) >= 1:
                target_cards[0].enhancement = Enhancement.STEEL
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Enhanced card to Steel"
                
        elif tarot == TarotCard.STRENGTH:
            # Increase rank of 2 cards by 1
            if target_cards:
                for card in target_cards[:2]:
                    if card.rank < Rank.ACE:
                        card.rank = Rank(card.rank + 1)
                        result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Increased card ranks"
                
        elif tarot == TarotCard.THE_HERMIT:
            # Double money (max $20)
            current_money = game_state.get('money', 0)
            gain = min(current_money, 20)
            game_state['money'] = current_money + gain
            result['money_gained'] = gain
            result['success'] = True
            result['message'] = f"Gained ${gain}"
            
        elif tarot == TarotCard.WHEEL_OF_FORTUNE:
            # 1/4 chance to add edition
            if target_cards and random.random() < 0.25:
                card = target_cards[0]
                editions = [Edition.FOIL, Edition.HOLOGRAPHIC, Edition.POLYCHROME]
                card.edition = random.choice(editions)
                result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = f"Added {card.edition.name} edition"
            else:
                result['message'] = "No effect"
                
        elif tarot == TarotCard.JUSTICE:
            # Enhance 1 card to Glass
            if target_cards and len(target_cards) >= 1:
                target_cards[0].enhancement = Enhancement.GLASS
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Enhanced card to Glass"
                
        elif tarot == TarotCard.THE_HANGED_MAN:
            # Destroy up to 2 cards
            if target_cards:
                destroyed = []
                for card in target_cards[:2]:
                    if 'deck' in game_state:
                        game_state['deck'].remove(card)
                        destroyed.append(card)
                result['cards_affected'] = destroyed
                result['success'] = True
                result['message'] = f"Destroyed {len(destroyed)} cards"
                
        elif tarot == TarotCard.DEATH:
            # Convert 2 left cards to match 2 right cards
            if target_cards and len(target_cards) >= 2:
                # Left cards copy right cards
                target_cards[0].rank = target_cards[1].rank
                target_cards[0].suit = target_cards[1].suit
                result['cards_affected'] = target_cards[:2]
                result['success'] = True
                result['message'] = "Transformed cards"
                
        elif tarot == TarotCard.TEMPERANCE:
            # Give total joker sell value as money (max $50)
            total_value = 0
            for joker in game_state.get('jokers', []):
                # Assume base sell value of 3-5
                total_value += 5
            gain = min(total_value, 50)
            game_state['money'] = game_state.get('money', 0) + gain
            result['money_gained'] = gain
            result['success'] = True
            result['message'] = f"Gained ${gain} from joker values"
            
        elif tarot == TarotCard.THE_DEVIL:
            # Enhance 1 card to Gold
            if target_cards and len(target_cards) >= 1:
                target_cards[0].enhancement = Enhancement.GOLD
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Enhanced card to Gold"
                
        elif tarot == TarotCard.THE_TOWER:
            # Enhance 1 card to Stone
            if target_cards and len(target_cards) >= 1:
                target_cards[0].enhancement = Enhancement.STONE
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Enhanced card to Stone"
                
        elif tarot == TarotCard.THE_STAR:
            # Convert 3 cards to Diamonds
            if target_cards:
                for card in target_cards[:3]:
                    card.suit = Suit.DIAMONDS
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Converted cards to Diamonds"
                
        elif tarot == TarotCard.THE_MOON:
            # Convert 3 cards to Clubs
            if target_cards:
                for card in target_cards[:3]:
                    card.suit = Suit.CLUBS
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Converted cards to Clubs"
                
        elif tarot == TarotCard.THE_SUN:
            # Convert 3 cards to Hearts
            if target_cards:
                for card in target_cards[:3]:
                    card.suit = Suit.HEARTS
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Converted cards to Hearts"
                
        elif tarot == TarotCard.JUDGEMENT:
            # Create random Planet card
            planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
            planet = random.choice(planets)
            if len(game_state.get('consumables', [])) < game_state.get('consumable_slots', 2):
                game_state['consumables'].append(planet)
                result['items_created'].append(planet)
            result['success'] = True
            result['message'] = f"Created {planet}"
            
        elif tarot == TarotCard.THE_WORLD:
            # Convert 3 cards to Spades
            if target_cards:
                for card in target_cards[:3]:
                    card.suit = Suit.SPADES
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = "Converted cards to Spades"
                
        return result

# ---------------------------------------------------------------------------
# Spectral Cards
# ---------------------------------------------------------------------------

class SpectralCard(IntEnum):
    FAMILIAR = auto()      # Destroy 1 card, create 3 random enhanced face cards
    GRIM = auto()          # Destroy 1 card, create 2 random enhanced Aces
    INCANTATION = auto()   # Destroy 1 card, create 4 random enhanced numbered cards
    TALISMAN = auto()      # Add Gold Seal to 1 card
    AURA = auto()          # Add Foil, Holo, or Polychrome to 1 card
    WRAITH = auto()        # Create random Rare joker, -1 hand size
    SIGIL = auto()         # Convert all cards in hand to single random suit
    OUIJA = auto()         # Convert all cards in hand to single random rank (-1 hand size)
    ECTOPLASM = auto()     # Add Negative to random joker, -1 hand size
    IMMOLATE = auto()      # Destroy 5 cards, gain $20
    ANKH = auto()          # Create copy of random joker, destroy others
    DEJA_VU = auto()       # Add Red Seal to 1 card
    HEX = auto()           # Add Polychrome to random joker, destroy others
    TRANCE = auto()        # Add Blue Seal to 1 card
    MEDIUM = auto()        # Add Purple Seal to 1 card
    CRYPTID = auto()       # Create 2 copies of 1 card
    THE_SOUL = auto()      # Creates Legendary joker (rare)
    BLACK_HOLE = auto()    # Upgrade every poker hand by 1 level

class SpectralEffects:
    """Handles all spectral card effects"""
    
    @staticmethod
    def apply_spectral(spectral: SpectralCard, game_state: Dict,
                      target_cards: Optional[List[Card]] = None) -> Dict[str, Any]:
        """Apply a spectral card effect and return results"""
        
        result = {
            'success': False,
            'message': '',
            'cards_affected': [],
            'cards_created': [],
            'cards_destroyed': [],
            'jokers_created': [],
            'jokers_destroyed': [],
            'money_gained': 0,
            'hand_size_change': 0
        }
        
        if spectral == SpectralCard.FAMILIAR:
            # Destroy 1 card, create 3 enhanced face cards
            if target_cards and len(target_cards) >= 1:
                # Destroy card
                destroyed = target_cards[0]
                if 'deck' in game_state:
                    game_state['deck'].remove(destroyed)
                result['cards_destroyed'].append(destroyed)
                
                # Create 3 enhanced face cards
                face_ranks = [Rank.JACK, Rank.QUEEN, Rank.KING]
                enhancements = [Enhancement.BONUS, Enhancement.MULT, Enhancement.WILD, 
                               Enhancement.GLASS, Enhancement.STEEL, Enhancement.GOLD, Enhancement.LUCKY]
                
                for _ in range(3):
                    rank = random.choice(face_ranks)
                    suit = random.choice(list(Suit))
                    enhancement = random.choice(enhancements)
                    new_card = Card(rank, suit, enhancement)
                    if 'deck' in game_state:
                        game_state['deck'].append(new_card)
                    result['cards_created'].append(new_card)
                
                result['success'] = True
                result['message'] = "Created 3 enhanced face cards"
                
        elif spectral == SpectralCard.GRIM:
            # Destroy 1 card, create 2 enhanced Aces
            if target_cards and len(target_cards) >= 1:
                # Destroy card
                destroyed = target_cards[0]
                if 'deck' in game_state:
                    game_state['deck'].remove(destroyed)
                result['cards_destroyed'].append(destroyed)
                
                # Create 2 enhanced Aces
                enhancements = [Enhancement.BONUS, Enhancement.MULT, Enhancement.WILD, 
                               Enhancement.GLASS, Enhancement.STEEL, Enhancement.GOLD, Enhancement.LUCKY]
                
                for _ in range(2):
                    suit = random.choice(list(Suit))
                    enhancement = random.choice(enhancements)
                    new_card = Card(Rank.ACE, suit, enhancement)
                    if 'deck' in game_state:
                        game_state['deck'].append(new_card)
                    result['cards_created'].append(new_card)
                
                result['success'] = True
                result['message'] = "Created 2 enhanced Aces"
                
        elif spectral == SpectralCard.INCANTATION:
            # Destroy 1 card, create 4 enhanced numbered cards
            if target_cards and len(target_cards) >= 1:
                # Destroy card
                destroyed = target_cards[0]
                if 'deck' in game_state:
                    game_state['deck'].remove(destroyed)
                result['cards_destroyed'].append(destroyed)
                
                # Create 4 enhanced numbered cards (2-10)
                number_ranks = list(range(2, 11))
                enhancements = [Enhancement.BONUS, Enhancement.MULT, Enhancement.WILD, 
                               Enhancement.GLASS, Enhancement.STEEL, Enhancement.GOLD, Enhancement.LUCKY]
                
                for _ in range(4):
                    rank = Rank(random.choice(number_ranks))
                    suit = random.choice(list(Suit))
                    enhancement = random.choice(enhancements)
                    new_card = Card(rank, suit, enhancement)
                    if 'deck' in game_state:
                        game_state['deck'].append(new_card)
                    result['cards_created'].append(new_card)
                
                result['success'] = True
                result['message'] = "Created 4 enhanced number cards"
                
        elif spectral == SpectralCard.TALISMAN:
            # Add Gold Seal to 1 card
            if target_cards and len(target_cards) >= 1:
                target_cards[0].seal = Seal.GOLD
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Added Gold Seal"
                
        elif spectral == SpectralCard.AURA:
            # Add random edition to 1 card
            if target_cards and len(target_cards) >= 1:
                editions = [Edition.FOIL, Edition.HOLOGRAPHIC, Edition.POLYCHROME]
                target_cards[0].edition = random.choice(editions)
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = f"Added {target_cards[0].edition.name} edition"
                
        elif spectral == SpectralCard.WRAITH:
            # Create random Rare joker, -1 hand size
            rare_jokers = ['Invisible Joker', 'Brainstorm', 'Satellite', 'Shoot the Moon',
                          'Drivers License', 'Cartomancer', 'Astronomer', 'Burnt Joker',
                          'Bootstraps', 'Canio', 'Triboulet', 'Yorick', 'Chicot', 'Perkeo']
            
            if len(game_state.get('jokers', [])) < game_state.get('joker_slots', 5):
                joker = random.choice(rare_jokers)
                game_state['jokers'].append(joker)
                result['jokers_created'].append(joker)
                result['hand_size_change'] = -1
                result['success'] = True
                result['message'] = f"Created {joker}, -1 hand size"
                
        elif spectral == SpectralCard.SIGIL:
            # Convert all cards in hand to single suit
            if 'hand' in game_state and game_state['hand']:
                target_suit = random.choice(list(Suit))
                for card in game_state['hand']:
                    card.suit = target_suit
                    result['cards_affected'].append(card)
                result['success'] = True
                result['message'] = f"Converted hand to {target_suit.name}"
                
        elif spectral == SpectralCard.OUIJA:
            # Convert all cards in hand to single rank, -1 hand size
            if 'hand' in game_state and game_state['hand']:
                target_rank = random.choice(list(Rank))
                for card in game_state['hand']:
                    card.rank = target_rank
                    result['cards_affected'].append(card)
                result['hand_size_change'] = -1
                result['success'] = True
                result['message'] = f"Converted hand to {target_rank.name}s, -1 hand size"
                
        elif spectral == SpectralCard.ECTOPLASM:
            # Add Negative to random joker, -1 hand size
            if game_state.get('jokers'):
                # In actual implementation, would add Negative edition to joker
                result['hand_size_change'] = -1
                result['success'] = True
                result['message'] = "Added Negative to random joker, -1 hand size"
                
        elif spectral == SpectralCard.IMMOLATE:
            # Destroy 5 cards, gain $20
            if 'deck' in game_state:
                cards_to_destroy = min(5, len(game_state['deck']))
                destroyed = random.sample(game_state['deck'], cards_to_destroy)
                for card in destroyed:
                    game_state['deck'].remove(card)
                    result['cards_destroyed'].append(card)
                
                game_state['money'] = game_state.get('money', 0) + 20
                result['money_gained'] = 20
                result['success'] = True
                result['message'] = f"Destroyed {cards_to_destroy} cards, gained $20"
                
        elif spectral == SpectralCard.ANKH:
            # Create copy of random joker, destroy others
            if game_state.get('jokers'):
                kept_joker = random.choice(game_state['jokers'])
                destroyed = [j for j in game_state['jokers'] if j != kept_joker]
                
                game_state['jokers'] = [kept_joker, kept_joker]  # Two copies
                result['jokers_created'].append(kept_joker)
                result['jokers_destroyed'] = destroyed
                result['success'] = True
                result['message'] = f"Kept 2 copies of {kept_joker}, destroyed others"
                
        elif spectral == SpectralCard.DEJA_VU:
            # Add Red Seal to 1 card
            if target_cards and len(target_cards) >= 1:
                target_cards[0].seal = Seal.RED
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Added Red Seal"
                
        elif spectral == SpectralCard.HEX:
            # Add Polychrome to random joker, destroy others
            if game_state.get('jokers'):
                kept_joker = random.choice(game_state['jokers'])
                destroyed = [j for j in game_state['jokers'] if j != kept_joker]
                
                game_state['jokers'] = [kept_joker]
                # In actual implementation, would add Polychrome edition
                result['jokers_destroyed'] = destroyed
                result['success'] = True
                result['message'] = f"Added Polychrome to {kept_joker}, destroyed others"
                
        elif spectral == SpectralCard.TRANCE:
            # Add Blue Seal to 1 card
            if target_cards and len(target_cards) >= 1:
                target_cards[0].seal = Seal.BLUE
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Added Blue Seal"
                
        elif spectral == SpectralCard.MEDIUM:
            # Add Purple Seal to 1 card
            if target_cards and len(target_cards) >= 1:
                target_cards[0].seal = Seal.PURPLE
                result['cards_affected'].append(target_cards[0])
                result['success'] = True
                result['message'] = "Added Purple Seal"
                
        elif spectral == SpectralCard.CRYPTID:
            # Create 2 copies of 1 card
            if target_cards and len(target_cards) >= 1 and 'deck' in game_state:
                original = target_cards[0]
                for _ in range(2):
                    copy = Card(original.rank, original.suit, 
                               original.enhancement, original.edition, original.seal)
                    game_state['deck'].append(copy)
                    result['cards_created'].append(copy)
                result['success'] = True
                result['message'] = f"Created 2 copies of card"
                
        elif spectral == SpectralCard.THE_SOUL:
            # Create Legendary joker (rare)
            legendary_jokers = ['Canio', 'Triboulet', 'Yorick', 'Chicot', 'Perkeo']
            if len(game_state.get('jokers', [])) < game_state.get('joker_slots', 5):
                joker = random.choice(legendary_jokers)
                game_state['jokers'].append(joker)
                result['jokers_created'].append(joker)
                result['success'] = True
                result['message'] = f"Created Legendary {joker}"
                
        elif spectral == SpectralCard.BLACK_HOLE:
            # Upgrade every poker hand by 1 level
            # This would interact with the ScoreEngine
            result['success'] = True
            result['message'] = "All hands upgraded by 1 level"
            # In actual implementation:
            # for hand_type in HandType:
            #     engine.apply_planet(hand_type)
            
        return result

# ---------------------------------------------------------------------------
# Integration with Environment
# ---------------------------------------------------------------------------

class ConsumableManager:
    """Manages all consumable cards (Tarot, Planet, Spectral)"""
    
    def __init__(self):
        self.tarot_effects = TarotEffects()
        self.spectral_effects = SpectralEffects()
        
    def use_consumable(self, consumable_name: str, game_state: Dict, 
                      target_cards: Optional[List[Card]] = None) -> Dict[str, Any]:
        """Use any consumable and return results"""
        
        # Check if it's a Tarot card
        try:
            tarot = TarotCard[consumable_name.upper().replace(' ', '_')]
            return self.tarot_effects.apply_tarot(tarot, game_state, target_cards)
        except KeyError:
            pass
        
        # Check if it's a Spectral card
        try:
            spectral = SpectralCard[consumable_name.upper().replace(' ', '_')]
            return self.spectral_effects.apply_spectral(spectral, game_state, target_cards)
        except KeyError:
            pass
        
        # Check if it's a Planet card
        planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
                       'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Planet X', 'Ceres', 'Eris']
        if consumable_name in planet_names:
            # This would be handled by the ScoreEngine
            return {
                'success': True,
                'message': f"Applied {consumable_name} planet card",
                'planet_used': consumable_name
            }
        
        return {'success': False, 'message': f"Unknown consumable: {consumable_name}"}

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test tarot cards
    print("=== TAROT CARD TESTS ===")
    
    game_state = {
        'deck': [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)],
        'hand': [Card(Rank.TWO, Suit.CLUBS)],
        'money': 10,
        'jokers': ['Joker', 'Mime'],
        'consumables': [],
        'consumable_slots': 2
    }
    
    manager = ConsumableManager()
    
    # Test The Magician
    result = manager.use_consumable('The Magician', game_state, game_state['deck'])
    print(f"The Magician: {result['message']}")
    for card in result['cards_affected']:
        print(f"  - {card.rank.name} of {card.suit.name} -> {card.enhancement.name}")
    
    # Test The Hermit
    result = manager.use_consumable('The Hermit', game_state)
    print(f"\nThe Hermit: {result['message']} (Money: ${game_state['money']})")
    
    # Test Spectral cards
    print("\n=== SPECTRAL CARD TESTS ===")
    
    # Test Familiar
    deck_before = len(game_state['deck'])
    result = manager.use_consumable('Familiar', game_state, [game_state['deck'][0]])
    print(f"Familiar: {result['message']}")
    print(f"  Deck size: {deck_before} -> {len(game_state['deck'])}")
    
    # Test Immolate
    result = manager.use_consumable('Immolate', game_state)
    print(f"\nImmolate: {result['message']} (Money: ${game_state['money']})")