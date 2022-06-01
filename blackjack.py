from CardDeck import *

actions = ['stand', 'hit', 'double', 'surrender']
actions_to_numbers = {'stand': 0, 'hit': 1, 'double': 2, 'surrender': 3}

class Blackjack:
    def __init__(self):
        self.action_space = 4
        self.observation_space = 3
        self.player_hand = []
        self.dealer_hand = []
        self.new_hand = False
        self.game_over = False
        self.is_soft = False
        self.deck = CardDeck(4)
        self.deck.shuffle()

    def get_best_hand(self, hand, who="dealer"):
        n_aces = 0
        sum = 0
        for card in hand:
            sum += card.getValue()
            if card.getValue() == 11:
                n_aces += 1
        while n_aces > 0 and sum > 21:
            sum -= 10
            n_aces -= 1
        if n_aces > 0 and who == "player":
            self.is_soft = True
        else:
            self.is_soft = False
        return sum

    def get_state(self):
        player_sum = self.get_best_hand(self.player_hand, "player")
        return [player_sum, self.dealer_hand[0].getValue(), int(self.is_soft)]

    def play_out_dealer(self):
        while self.get_best_hand(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.deal())
        return self.get_best_hand(self.dealer_hand)

    def is_tie(self):
        return(self.get_best_hand(self.player_hand, "player") == self.get_best_hand(self.dealer_hand))

    def is_natural_bj(self):
        if self.new_hand and self.get_best_hand(self.player_hand, "player") == 21:
            return True

    def reset(self):
        self.new_hand = True
        self.game_over = False
        self.player_hand = []
        self.dealer_hand = []
        for _ in range(2):
            if self.deck.isEmpty:
                self.deck.refill()
            self.dealer_hand.append(self.deck.deal())
            self.player_hand.append(self.deck.deal())
        return self.get_state()

    def get_reward(self, action):
        if action == "stand":
            self.game_over = True
            self.new_hand = False
            dealer_sum = self.play_out_dealer()
            player_sum = self.get_best_hand(self.player_hand, "player")
            if player_sum > 21:
                return -1
            if dealer_sum > 21:
                return 1
            if player_sum > dealer_sum:
                return 1
            if player_sum < dealer_sum:
                return -1
            else:
                return 0
            
        if action == "hit":
            self.new_hand = False
            self.player_hand.append(self.deck.deal())
            player_sum = self.get_best_hand(self.player_hand, "player")
            if player_sum > 21:
                self.game_over = True
                return -1
            if player_sum == 21:
                self.game_over = True
                if (self.is_tie() == True):
                    return 0
                return 1
            else:
                return 0

        if action == "double":
            self.player_hand.append(self.deck.deal())
            player_sum = self.get_best_hand(self.player_hand, "player")
            self.play_out_dealer()
            dealer_sum = self.get_best_hand(self.dealer_hand, "player")
            self.game_over = True
            if self.new_hand:
                self.new_hand = False
                if player_sum > 21:
                    return -2
                elif dealer_sum > 21:
                    return 2 
                if (self.is_tie() == True):
                        return 0
                elif dealer_sum > player_sum:
                    return -2
                elif dealer_sum < player_sum:
                    return 2
            else:
                print("impossible action was taken")
        
        if action == "surrender":
            self.game_over = True
            return - 0.5

    def step(self, action):
        assert (0 <= actions_to_numbers[action] <= self.action_space)

        if self.is_natural_bj():
            self.game_over = True
            reward = 1.5
        if self.get_best_hand(self.player_hand, "player") == 21:
            self.game_over = True
            reward = 1
        else:
            reward = self.get_reward(action)

        return self.get_state(), reward, self.game_over