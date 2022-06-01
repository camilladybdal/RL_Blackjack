import random

blackjackValues = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10, 'A':11}
suits = ['Hearts', 'Diamonds', 'Spades', 'Clubs']

class Card(object):
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.rank_int = blackjackValues[self.rank]

    def __str__(self):
        return self.rank + " of " + self.suit

    def getValue(self):
        return self.rank_int

class CardDeck(object):
    def __init__(self, n_decks):
        self.deck = []
        self.n_decks = n_decks
        self.buildDeck()
    
    def __str__(self): 
        str = ''
        for i in range(len(self.deck)):
            str += self.deck[i] + '\n'
            return str

    def buildDeck(self):
        for decks in range(self.n_decks):
            for suit in suits:
                for rank in blackjackValues:
                    self.deck.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        return self.deck.pop()
    
    def isEmpty(self):
        return len(self.deck) == 0
    
    def refill(self):
        self.deck = []
        self.buildDeck()
        self.shuffle()

def main():
    deck = CardDeck(1)
    deck.shuffle()
    for card in deck.deck:
        print(card)

if __name__ == "__main__":
    main()