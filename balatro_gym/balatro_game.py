import numpy as np

from enum import Enum

class Card:
	class Ranks(Enum):
		TWO = 0
		THREE = 1
		FOUR = 2
		FIVE = 3
		SIX = 4
		SEVEN = 5
		EIGHT = 6
		NINE = 7
		TEN = 8
		JACK = 9
		QUEEN = 10
		KING = 11
		ACE = 12

	base_chip_values = {
		Ranks.TWO: 2,
		Ranks.THREE: 3,
		Ranks.FOUR: 4,
		Ranks.FIVE: 5,
		Ranks.SIX: 6,
		Ranks.SEVEN: 7,
		Ranks.EIGHT: 8,
		Ranks.NINE: 9,
		Ranks.TEN: 10,
		Ranks.JACK: 10,
		Ranks.QUEEN: 10,
		Ranks.KING: 10,
		Ranks.ACE: 11,
	}

	class Suits(Enum):
		SPADES = 0
		CLUBS = 1
		HEARTS = 2
		DIAMONDS = 3

	def __init__(self, rank, suit):
		self.rank = rank
		self.suit = suit
		self.played = False

	def chip_value(self):
		return self.base_chip_values[self.rank]
	
	def __str__(self):
		return self.rank.name + " OF " + self.suit.name

# class PokerHands(Enum):
# 	HIGH_CARD = 0
# 	PAIR = 1
# 	TWO_PAIR = 2
# 	THREE_OF_A_KIND = 3
# 	STRAIGHT = 4
# 	FLUSH = 5
# 	FULL_HOUSE = 6
# 	FOUR_OF_A_KIND = 7
# 	STRAIGHT_FLUSH = 8
# 	ROYAL_FLUSH = 9
# 	FIVE_OF_A_KIND = 10
# 	FLUSH_HOUSE = 11
# 	FLUSH_FIVE = 12


class BalatroGame:
	class State(Enum):
		IN_PROGRESS = 0
		WIN = 1
		LOSS = 2

	class Stages(Enum):
		BLIND_SELECT = 0
		ROUND = 1
		SHOP = 2

	def __init__(self, deck="yellow", stake="white"):
		self.deck = []

		# Standard deck
		for suit in Card.Suits:
			for rank in Card.Ranks:
				self.deck.append(Card(rank, suit))

		self.hand = []
		self.selection = []

		self.hand_size = 8
		self.hands = 4
		self.discards = 3
		
		self.ante = 0
		self.blind_index = 0
		self.blinds = [300, 450, 600]

		self.stage = self.Stages.ROUND
		self.round_score = 0
		self.round_hands = self.hands
		self.round_discards = self.discards

		self.state = self.State.IN_PROGRESS

	def select_card(self, hand_index: int):
		self.selection.append(self.hand.pop(hand_index))

	def play_hand(self):
		self.round_hands -= 1

		chips = 0
		mult = 0
		scoring_cards = None

		played_cards = [self.deck[card_index] for card_index in self.selection]

		flush = len({card.suit for card in played_cards}) == 1
		straight = True

		sorted_ranks = sorted([card.rank.value for card in played_cards])
		if sorted_ranks != [0, 1, 2, 3, 12]:
			for i in range(len(sorted_ranks) - 1):
				if sorted_ranks[i] + 1 != sorted_ranks[i + 1]:
					straight = False
					break
		
		rank_counts = {14: [], 15: []}
		for card in played_cards:
			if card.rank not in rank_counts:
				rank_counts[card.rank] = [card]
			else:
				rank_counts[card.rank].append(card)

		primary_hand, secondary_hand = sorted(rank_counts.values(), key=lambda x: len(x), reverse=True)[0:2]

		if flush and len(primary_hand) == 5:
			chips += 160
			mult += 16
			scoring_cards = played_cards
		elif flush and len(primary_hand) == 3 and len(secondary_hand) == 2:
			chips += 140
			mult += 14
			scoring_cards = played_cards
		elif len(primary_hand) == 5:
			chips += 120
			mult += 12
			scoring_cards = played_cards
		elif straight and flush:
			chips += 100
			mult += 8
			scoring_cards = played_cards
		elif len(primary_hand) == 4:
			chips += 60
			mult += 7
			scoring_cards = primary_hand
		elif len(primary_hand) == 3 and len(secondary_hand) == 2:
			chips += 40
			mult += 4
			scoring_cards = played_cards
		elif flush:
			chips += 35
			mult += 4
			scoring_cards = played_cards
		elif straight:
			chips += 30
			mult += 4
			scoring_cards = played_cards
		elif len(primary_hand) == 3:
			chips += 30
			mult += 3
			scoring_cards = primary_hand
		elif len(primary_hand) == 2 and len(secondary_hand) == 2:
			chips += 20
			mult += 2
			scoring_cards = primary_hand + secondary_hand
		elif len(primary_hand) == 2:
			chips += 10
			mult += 2
			scoring_cards = primary_hand
		else:
			chips += 5
			mult += 1
			scoring_cards = [max(played_cards, key=lambda card: card.rank.value)]

		for card in scoring_cards:
			chips += card.chip_value()

		self.round_score += chips * mult
		print(chips * mult)

		if self.round_score >= self.blinds[self.blind_index]:
			self._end_round()
		elif self.round_hands == 0:
			self.state = self.State.LOSS
		else:
			self._draw_cards()

	def discard_hand(self):
		self.round_discards -= 1
		self._draw_cards()

	def _end_round(self):
		self.round_score = 0
		self.blind_index += 1
		self._reset_cards()

	def _draw_cards(self):
		self._reset_cards()
		remaining_cards = [i for i in range(len(self.deck)) if not self.deck[i].played]

		for card_index in np.random.choice(remaining_cards, min(self.hand_size - len(self.hand), len(remaining_cards)), replace=False):
			self.deck[card_index].played = True
			self.hand.append(card_index)

	def _reset_cards(self):
		for card in self.deck:
			card.played = False
		
		self.hand.clear()
		self.selection.clear()

	def print_deck(self):
		print([str(card) for card in self.deck])

	def print_hand(self):
		print([str(self.deck[card_index]) for card_index in self.hand])

	def print_selected(self):
		print([str(self.deck[card_index]) for card_index in self.selection])

		
