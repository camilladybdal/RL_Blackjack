class Blackjack:
        def __init__(self):
            self.action_space = 4 #hit, stand, double, surrender
            self.observation_space = 5  # [player_sum, self.is_soft, self.dealer_hand[0].get_value(), self.new_hand]
            self.actions = ['stand', 'hit', 'double', 'surrender']
            self.actions_to_numbers = {'stand': 0, 'hit': 1, 'double': 2, 'surrender': 3}
            self.player_hand = []
            self.dealer_hand = []
            self.new_hand = False
            self.game_over = False
            _ = self.reset()

def reset(self):
    self.new_hand = True
    self.game_over = False

    for _ in range(2):
        if self.deck.isEmpty:
            self.deck.refill()
        self.dealer_hand.append(self.deck.deal())
        self.player_hand.append(self.deck.deal())
            
    return self.get_state()

def get_best_hand(self, hand):
    n_aces = 0
    sum = 0
    soft = False
    for card in hand:
        sum += card.get_value()
        if card == 11:
            n_aces += 1
    while n_aces > 0 and sum > 21:
        sum -= 10
        n_aces -= 1
    if n_aces > 0 and hand == "self.player_hand":
        soft = True
    return sum , soft

def get_state(self):
    player_sum = self.get_best_hand(self.player_hand)
    if len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]:
        pair = True
    return [player_sum, self.is_soft, self.dealer_hand[0].get_value(), self.new_hand]


def play_out_dealer(self):
    while self.calculate_hand(self.dealer_hand) < 17:
        self.dealer_hand.append(self.deck.deal())
    return self.calculate_hand(self.dealer_hand)

def is_tie(self):
    return(self.get_best_hand(self.player_hand) == self.get_best_hand(self.dealer_hand))


def get_reward(self, action):
    if action == "stand":
        self.game_over = True
        self.new_hand = False
        dealer_sum = self.play_out_dealer()
        player_sum = self.get_best_hand(self.player_hand)
        if dealer_sum > 21:
            return -1
        if dealer_sum > 21:
            return 1
        if player_sum > dealer_sum:
            return 1
        if player_sum < dealer_sum:
            return -1
        
    if action == "hit":
        self.new_hand = False
        self.player_hand.append(self.deck.deal())
        player_sum = self.get_best_hand(self.player_hand)
        if player_sum > 21:
            self.game_over = True
            return -1
        return 0

    if action == "double":
        self.player_hand.append(self.deck.deal())
        player_sum = self.get_best_hand(self.player_hand)
        self.play_out_dealer()
        dealer_sum = self.get_best_hand(self.dealer_hand)
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
            return 0
    
    if action == "surrender":
        self.game_over = True
        return -0.5


def step(self, action):
    assert (0 <= action <= self.action_space)
    reward = self.get_reward(action)
    return self.get_state(), reward, self.game_over