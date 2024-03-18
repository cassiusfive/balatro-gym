from balatro_gym.balatro_game import BalatroGame, Card

def test_flush_five():
    hand = [Card(Card.Ranks.ACE, Card.Suits.SPADES) for _ in range(5)]
    assert BalatroGame._evaluate_hand(hand) == 3440
    hand = [Card(Card.Ranks.SIX, Card.Suits.DIAMONDS) for _ in range(5)]
    assert BalatroGame._evaluate_hand(hand) == 3040

def test_straight_flush():
    hand = [Card(Card.Ranks(i), Card.Suits.SPADES) for i in range(0, 5)]
    assert BalatroGame._evaluate_hand(hand) == 960
    hand = [Card(Card.Ranks(i), Card.Suits.DIAMONDS) for i in range(12, 7, -1)]
    assert BalatroGame._evaluate_hand(hand) == 1208
    hand = [Card(Card.Ranks(i), Card.Suits.CLUBS) for i in range(0, 4)]
    hand.append(Card(Card.Ranks.ACE, Card.Suits.CLUBS))
    assert BalatroGame._evaluate_hand(hand) == 1000

def test_straight():
    hand = [Card(Card.Ranks(i), Card.Suits.CLUBS) for i in range(0, 4)]
    hand.append(Card(Card.Ranks.ACE, Card.Suits.DIAMONDS))
    assert BalatroGame._evaluate_hand(hand) == 220
