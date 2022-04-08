import random
from select import select
import string
import sys
import time
from collections import defaultdict

import numpy as np

DICO_PATH = './dico.txt'
DICO_INST = []

def damerau_levenshtein_distance(s1, s2):
    """Fonction permettant de calculer la distance entre 2 chaines de characteres
    @pa
    """
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(d[(i-1,j)] + 1,
                           d[(i,j-1)] + 1,
                           d[(i-1,j-1)] + cost)
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)

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
        if word not in self.dico:
            print(f'Rejected invalid word guess: {word}')
            return 0, 0
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
                  f'Time taken: {np.ceil(self.endTime - self.startTime)}s')
        else:
            print(f'Failed to solve WordleMind in {self.nbTries} tries.\n'
                  f'Time taken: {np.ceil(self.endTime - self.startTime)}s')

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


class Genetics(WordleMindGuesser):
    def __init__(self, word_length, wordleMindInst : WordleMind, maxSize, maxGen = 25, timeout = 300,
                 probMut = 0.5, verbose = False):
        super().__init__(word_length, wordleMindInst, verbose)
        self.population = np.array([])
        self.children = np.array([])
        self.prev_words = []
        self.prev_green = []
        self.prev_orange = []
        self.maxSize = maxSize
        self.maxGen = maxGen
        self.goodGuesses = np.array([])
        self.timeout = timeout
        self.probMut = probMut

    def guess(self, strStart):
        
        chosenWord = random.choice(self.wordleMindInst.dico)
        greenLetters, orangeLetters = self.guessWordRegister(chosenWord)
        if self.verbose:
            print(f'Chosen word {chosenWord} - Green: {greenLetters} | Orange: {orangeLetters}')
        if greenLetters == self.word_length:
            return True
        self.initPop(100)
        while True:
            geneticStartTime = time.time()
            while len(self.population)==0 and time.time() < geneticStartTime + self.timeout :
                self.genetics()
                if len(self.population)>self.maxSize/2:
                    self.children = self.population
                else:
                    self.initPop(100)
                    self.children = np.append(self.children,self.population)
                    np.random.shuffle(self.children)
                

            if self.verbose and time.time() >= geneticStartTime + self.timeout:
                print('Genetic algorithm timed out.')

            chosenWord = self.choice()
            self.population = np.array([])
            greenLetters, orangeLetters = self.guessWordRegister(chosenWord)
            if self.verbose:
                print(f'Chosen word {chosenWord} - Green: {greenLetters} | Orange: {orangeLetters}')
            if greenLetters == self.word_length:
                self.solution = chosenWord
                return True
        return False

    def guessWordRegister(self, word):
        greenLetters, orangeLetters = self.wordleMindInst.checkWord(word)
        self.nbTries += 1
        self.prev_words.append(word)
        self.prev_green.append(greenLetters)
        self.prev_orange.append(orangeLetters)
        return greenLetters, orangeLetters

    def choice(self):
        return random.choice(self.population)

    def approximatePop(self):
        newPop = set()
        for word in self.children:
            bestWords = set()
            bestDist = np.inf
            for dictWord in self.wordleMindInst.dico:
                if dictWord == word:
                    newPop.add(word)
                    break
                curDist = damerau_levenshtein_distance(word, dictWord)
                if curDist <= bestDist:
                    bestDist = curDist
                    bestWords = set()
                    bestWords.add(dictWord)
                elif curDist == bestDist:
                    bestWords.add(dictWord)
            newPop = newPop.union(bestWords)
        popSet = set(newPop)
        self.children = np.array(list(popSet))
        return

    def fitness(self, word):
        value = 0
        for index in range(len(self.prev_words)):
            green, orange = compareWords(word, self.prev_words[index])
            if (green,orange) != (self.prev_green[index],self.prev_orange[index]):
                value += 1
        return value

    def ajout_Compatible(self):
        for word in self.children:
            if (word not in self.population.tolist()) and (self.fitness(word) == 0):
                self.population = np.append(self.population,word)
        return 

    def selection(self,n):
        to_cross = []
        for ind in self.children:
            fit = self.fitness(ind)
            if (np.random.random()+ fit/len(self.prev_words) < 1):
                to_cross.append(ind)

        return to_cross

    @staticmethod
    def mutateSwap(word):
        if len(word) == 1:
            return word
        toSwap = np.random.choice(range(len(word)), 2, replace=False)
        out = list(word)
        out[toSwap[1]], out[toSwap[0]] = out[toSwap[0]], out[toSwap[1]]
        return ''.join(out)

    @staticmethod
    def mutateSeq(word):
        if len(word) == 1:
            return  word
        cut = np.random.choice(range(1, len(word)))
        return word[cut:] + word[:cut]

    @staticmethod
    def mutateLetter(word):
        toSwap = np.random.choice(range(len(word)))
        newLetter = random.choice(string.ascii_lowercase)
        while newLetter == toSwap:
            newLetter = np.random.choice(string.ascii_lowercase)
        return word[:toSwap] + newLetter + word[toSwap + 1:]

    @staticmethod
    def crossHalf(p1, p2):
        cutOff = len(p1) // 2
        out1 = p2[:cutOff] + p1[cutOff:]
        out2 = p1[:cutOff] + p2[cutOff:]
        return out1, out2

    @staticmethod
    def cross(to_cross:list):
        childrenCross = []
        while len(childrenCross) < 30:
            childrenCross.extend(Genetics.crossHalf(np.random.choice(childrenCross,2,replace = False)))
        return childrenCross

    def generateRandomString(self):
        return ''.join(random.sample(string.ascii_lowercase, self.word_length))

    def initPop(self,n):
        popSet = set()
        while len(popSet) < n:
            popSet.add(self.generateRandomString())
        self.children = np.array(list(popSet))
        self.approximatePop()

    def genetics(self):
        curGen = 0
        while curGen < self.maxGen :
            if self.verbose:
                print(f'----- GENERATION {curGen} Pop: {len(self.population)}- Children: {len(self.children)}-----')
            self.ajout_Compatible()
            to_cross = self.selection(self.maxSize/2)

            if len(self.population)>self.maxSize:
                print(f'MAX SIZE REACHED')
                break

            #crossRange = range(1, len(to_cross), 2)
            #childrenCross = []
            #for i in crossRange:
            #    childrenCross.extend(self.crossHalf(to_cross[i], to_cross[i-1]))
            childrenCross = self.cross(to_cross)
            childrenMut = []
            for ind in childrenCross:
                childMut = ind
                for mutate in [self.mutateSeq, self.mutateSwap, self.mutateLetter]:
                    if np.random.random() < self.probMut:
                        childMut = mutate(childMut)
                childrenMut.append(childMut)
            self.children = np.array(childrenCross)
            self.children = np.append(self.children,childrenMut)
            #childrenList = list(self.children)
            #childrenList.extend(childrenMut)
            self.approximatePop()

            curGen += 1
            if self.verbose and curGen >= self.maxGen:
                print('Genetic algorithm ran out of generations')
                return
        if self.verbose:
            print(f'------------ DONE WITH GENETICS ------------')
        return 

def main(argv):
    global DICO_INST
    secret = "dirty"
    word_length = len(secret)
    """
    try:
        word_length = int(argv[1])
    except IndexError:
        print(f'Not enough arguments given, expected 2 got {len(argv)}\nFormat: python3 ./main.py <word_length>')
    except ValueError:
        print('Invalid word length given as parameter')
    """
    DICO_INST = loadDico(DICO_PATH, word_length)

    verbose = True
    wMind = WordleMind(DICO_INST, secret)

    """
    bChrono = BackTrackChrono(word_length , wMind, verbose)
    bChrono.startGuessing()

    bChronoArc = BackTrackChronoArc(word_length , wMind, verbose)
    bChronoArc.startGuessing()

    bChrono.results()
    bChronoArc.results()"""

    genGuesser = Genetics(word_length, wMind, 100, verbose=verbose)
    genGuesser.startGuessing('')
    genGuesser.results()



if __name__ == '__main__':
    main(sys.argv)

