import random
import sys
import time
from collections import defaultdict

DICO_PATH = './dico.txt'
DICO_INST = []

GREEN_WEIGHT = 5
ORANGE_WEIGHT = 1

def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1,
                           d[(i,j-1)] + 1,
                           d[(i-1,j-1)] + cost,
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost)

    return d[lenstr1-1,lenstr2-1]

def loadDico(path, n):
    with open(path,'r') as f:
        return [line.rstrip('\n') for line in f if len(line.rstrip('\n')) == n]

def compareWords(word, other):
    # Right letter right place
    greenLetters = 0
    # Right letter wrong place
    orangeLetters = 0

    remainWord = defaultdict(lambda: 0)
    remainOther = defaultdict(lambda: 0)

    for i in range(len(word)):
        if word[i] == other[i]:
            greenLetters += 1
        else:
            remainWord[word[i]] += 1
            remainOther[other[i]] += 1

    for i in remainWord:
        if remainOther[i] != 0:
            orangeLetters += min(remainWord[i], remainOther[i])

    return greenLetters, orangeLetters

class WordleMind:
    def __init__(self, dico, secret = None):
        self.dico = dico
        if secret is None:
            self.secret = random.choice(self.dico)
        else:
            self.secret = secret

    def testDicoStart(self, s):
        for word in self.dico:
            if word.startswith(s):
                return True
        return False

    def checkWord(self, word):
        return compareWords(word, self.secret)

class WordleMindGuesser:
    def __init__(self, word_length, wordleMindInst : WordleMind, verbose = False):
        self.word_length = word_length
        self.nbTries = 0
        self.startTime = None
        self.endTime = None
        self.wordleMindInst = wordleMindInst
        self.solution = None
        self.verbose = verbose
        self.guessed = False

    def startGuessing(self, strStart = ''):
        self.startTime = time.time()
        self.guessed = self.guess(strStart)
        self.endTime = time.time()

    def guess(self, strStart):
        raise NotImplementedError

    def results(self):
        if self.guessed:
            print(f'WordleMind solved with word {self.solution} in {self.nbTries} tries.\n'
                  f'Time taken: {self.endTime - self.startTime}s')
        else:
            print(f'Failed to solve WordleMind in {self.nbTries} tries.\n'
                  f'Time taken: {self.endTime - self.startTime}s')

class BackTrackChrono(WordleMindGuesser):
    def __init__(self, word_length, wordleMindInst: WordleMind, verbose=False):
        super().__init__(word_length, wordleMindInst, verbose)

    def guess(self, strStart):
        for i in range(26):
            nextStr = strStart + chr((ord('a') + i))
            if self.wordleMindInst.testDicoStart(nextStr):
                if len(nextStr) == self.word_length:
                    if self.verbose:
                        print(f'BackTrackChrono: Trying {nextStr}')
                    green, _ = self.wordleMindInst.checkWord(nextStr)
                    self.nbTries += 1
                    if green == self.word_length:
                        self.solution = nextStr
                        return True
                else:
                    if self.guess(nextStr):
                        return True
        return False

class BackTrackChronoArc(WordleMindGuesser):
    def __init__(self, word_length, wordleMindInst: WordleMind, verbose=False):
        super().__init__(word_length, wordleMindInst, verbose)
        self.bestWord = ''
        self.bestScore = -1

    def guess(self, strStart):
        for i in range(26):
            nextStr = strStart + chr((ord('a') + i))
            if self.wordleMindInst.testDicoStart(nextStr):
                if len(nextStr) == self.word_length and self.checkCompat(nextStr):
                    if self.verbose:
                        print(f'BackTrackChronoArc: Trying {nextStr}')
                    green, orange = self.wordleMindInst.checkWord(nextStr)
                    self.nbTries += 1

                    if green == self.word_length:
                        self.solution = nextStr
                        return True
                    if green + orange > self.bestScore:
                        self.bestScore = green + orange
                        self.bestWord = nextStr
                else:
                    if self.guess(nextStr):
                        return True
        return False

    def checkCompat(self, word):
        if self.bestScore == -1:
            return True
        green, orange = compareWords(word, self.bestWord)
        if green + orange >= self.bestScore :
            return True
        return False


class Genetics:
    
    def __init__(self, word_length,wordleMindInst:WordleMind,verbose = False ):
        super().__init__(word_length, wordleMindInst, verbose)
        self.prev_words = []
        self.prev_green =[]
        self.prev_orange =[]

    def fitness(word):
        value = 0
        for index in range(len(self.prev_words)):
            green, orange = compareWords(word, prev_word[index])
            if (green,orange) != (self.prev_green[index],self.prev_orange[index]):
                value +=1
        return value

    def genetics(filtered_dico, word_length, maxgen, maxsize, timeout):
        start_time = time.time()


def main(argv):
    global DICO_INST
    word_length = -1
    try:
        word_length = int(argv[1])
    except IndexError:
        print(f'Not enough arguments given, expected 2 got {len(argv)}\nFormat: python3 ./main.py <word_length>')
    except ValueError:
        print('Invalid word length given as parameter')
    DICO_INST = loadDico(DICO_PATH, word_length)

    verbose = True

    wMind = WordleMind(DICO_INST, 'dirty')
    bChrono = BackTrackChrono(word_length , wMind, verbose)
    bChrono.startGuessing()

    bChronoArc = BackTrackChronoArc(word_length , wMind, verbose)
    bChronoArc.startGuessing()

    bChrono.results()
    bChronoArc.results()

if __name__ == '__main__':
    main(sys.argv)

