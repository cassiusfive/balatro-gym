from balatro_gym.balatro_game import BalatroGame

balatro = BalatroGame()

balatro._draw_cards()

balatro.selection = [0, 1, 2, 3, 4]
balatro.print_selected()
balatro.play_hand()



