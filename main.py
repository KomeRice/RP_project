import random
import string
import sys
import time
from collections import defaultdict

import argparse
import numpy as np
import pandas as pd
DICO_PATH = './dico.txt'
DICO_INST = []
DICO = []

with open(DICO_PATH,'r') as f:
    DICO = [line.rstrip('\n') for line in f]

def distance(s1,s2):
    v = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            v+=1
    return v

def loadDico(path, n):
    """Charge le dictionnaire
    """
    with open(path,'r') as f:
        return [line.rstrip('\n') for line in f if len(line.rstrip('\n')) == n]

def compareWords(word, other):
    """Evalue le nombre de lettre a la bonne place et de lettre present mais mal placé entre deux mots
    @return greenLetter: le nombre de lettre a la bonne place 
    @return orangeLetter : le nombre de lettre present mais mal placé
    """
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
    """Represente une instance du jeu
    """
    def __init__(self, dico, secret = None):
        self.dico = dico
        if secret is None:
            self.secret = random.choice(self.dico)
        else:
            self.secret = secret

    def testDicoStart(self, s):
        """Teste si un mot dans le dictionnaire commence avec le string s
        """
        for word in self.dico:
            if word.startswith(s):
                return True
        return False

    def checkWord(self, word):
        """Teste un mot choisi
        """
        if word not in self.dico:
            print(f'Rejected invalid word guess: {word}')
            return 0, 0
        return compareWords(word, self.secret)

class WordleMindGuesser:
    """Class mere des instance de decouverte de mot caché
    """
    def __init__(self, word_length, wordleMindInst : WordleMind, verbose = False):
        self.word_length = word_length #taille du mot a deviner
        self.nbTries = 0 #nombre d'essai
        self.startTime = None   #temps de depart
        self.endTime = None     #temps de fin
        self.wordleMindInst = wordleMindInst    #instance de jeu 
        self.solution = None    #solution supposé
        self.verbose = verbose  
        self.guessed = False    #mot deviné ou pas

    def startGuessing(self, strStart = ''):
        """Lance la recherce
        """
        self.startTime = time.time()
        self.guessed = self.guess(strStart)
        self.endTime = time.time()

    def guess(self, strStart):
        """Abstrait : recherche d'ensemble de mot compatible
        """
        raise NotImplementedError

    def results(self):
        """Affiche le resultat de la recherche
        """
        if self.guessed:
            print(f'WordleMind solved with word {self.solution} in {self.nbTries} tries.\t'
                  f'Time taken: {np.ceil(self.endTime - self.startTime)}s')
            return 1,self.endTime - self.startTime , self.nbTries
        else:
            print(f'Failed to solve WordleMind in {self.nbTries} tries.\t'
                  f'Time taken: {np.ceil(self.endTime - self.startTime)}s')
            return 0,self.endTime - self.startTime , self.nbTries

class BackTrackChrono(WordleMindGuesser):
    """ Generation de mot en utilisant le backtraking
    """
    def __init__(self, word_length, wordleMindInst: WordleMind, verbose=False):
        super().__init__(word_length, wordleMindInst, verbose)

    def guess(self, strStart):
        #On commence par A, si un mot dans le dictionnaire commence par cette lettre on ajoute un autre A, sinon on passe a B
        #Cette logique est iteré jusqu'a ce que l'on trouve le mot ou que l'on arrive a Z fois la taille du mot
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
    """Generation de mot en utilisant le backtraking avec coherence avec les mots evalués
    """
    def __init__(self, word_length, wordleMindInst: WordleMind, verbose=False):
        super().__init__(word_length, wordleMindInst, verbose)
       # self.bestWord = '' #meileur mot deviné
        #self.bestScore = -1 #score du meilleur mot deviné
        self.prev_words = []
        self.prev_green = []
        self.prev_orange = []

    def guess(self, strStart):
        #Meme principe que BackTrackChrono mais entre chaque essai de mot, on verifie la compatibilite avec les mots deja essayé
        for i in range(26):
            nextStr = strStart + chr((ord('a') + i))
            if self.wordleMindInst.testDicoStart(nextStr):
                if len(nextStr) == self.word_length and self.checkCompat(nextStr):
                    if self.verbose:
                        print(f'BackTrackChronoArc: Trying {nextStr}')
                    green, orange = self.wordleMindInst.checkWord(nextStr)
                    self.nbTries += 1
                    self.prev_words.append(nextStr)
                    self.prev_green.append(green)
                    self.prev_orange.append(orange)

                    if green == self.word_length:
                        self.solution = nextStr
                        return True
                    #if green + orange > self.bestScore:
                    #    self.bestScore = green + orange
                    #    self.bestWord = nextStr
                    
                else:
                    if self.guess(nextStr):
                        return True
        return False

    def checkCompat(self, word):
        """Verifie la compatibilité du mot avec ceux deja essayé
        """
        for ind in range(len(self.prev_words)):
            green, orange = compareWords(word, self.prev_words[ind])
            if (green,orange) != (self.prev_green[ind],self.prev_orange[ind]):
                return False
        return True

class Genetics(WordleMindGuesser):
    """Generation de mot en utilisant un algoritme genetique
    """
    def __init__(self, word_length, wordleMindInst : WordleMind, maxSize = 50, maxGen = 10, timeout = 300,
                 probMut = 0.3, verbose = False):
        super().__init__(word_length, wordleMindInst, verbose)
        self.population = np.array([]) #ensemble de mot compatible
        self.children = np.array([]) #ensemble de mot generé
        self.prev_words = [] #mots deja evalué
        self.prev_green = [] #nombre de bonne lettre des mots evalués
        self.prev_orange = [] #nombre de lettre mal placé des mots evalués
        self.maxSize = maxSize #taille maximum de l'ensemble de mots compatible
        self.maxGen = maxGen # nombre maximum de generation
        self.timeout = timeout #temps limite pour trouver des mots compatibles
        self.probMut = probMut #probabilité de mutation

    def guess(self, strStart):
        """Utilisation d'un algoritme genetique pour trouver une liste de mots compatibles puis choix d'un mot au hasard
        jusqu'a trouver le mot recherché ou atteindre le time out
        """
        #premier mot choisi au hasard
        chosenWord = random.choice(self.wordleMindInst.dico)
        greenLetters, orangeLetters = self.guessWordRegister(chosenWord)
        if self.verbose:
            print(f'Chosen word {chosenWord} - Green: {greenLetters} | Orange: {orangeLetters}')
        if greenLetters == self.word_length:
            return True
        #initialiser une liste de mots
        self.initPop(200)
        while True:
            geneticStartTime = time.time()
            while len(self.population)==0 and time.time() < geneticStartTime + self.timeout :
                self.genetics()
                #on reprend la liste des mots compatible d'avant pour faire les croissement
                if len(self.population)>self.maxSize/2:
                    self.children = self.population
                else:
                    #si pas assez de mots : on genere d'autre mots aleatoire
                    self.initPop(200)
                    self.children = np.append(self.children,self.population)
                    np.random.shuffle(self.children)
            #Temps de recherche de mot expiré
            if time.time() >= geneticStartTime + self.timeout:
                if self.verbose:
                    print('Genetic algorithm timed out.')
                if len(self.population) == 0:
                    return False
            #Choix d'un mot parmis les compatibles
            chosenWord = self.choice()
            self.population = np.array([])
            #evaluation de ce mot
            greenLetters, orangeLetters = self.guessWordRegister(chosenWord)
            if self.verbose:
                print(f'Chosen word {chosenWord} - Green: {greenLetters} | Orange: {orangeLetters}')
            if greenLetters == self.word_length:
                #Solution trouvé
                self.solution = chosenWord
                return True

    def guessWordRegister(self, word):
        """Enregistre un mot evalué
        """
        greenLetters, orangeLetters = self.wordleMindInst.checkWord(word)
        self.nbTries += 1
        self.prev_words.append(word)
        self.prev_green.append(greenLetters)
        self.prev_orange.append(orangeLetters)
        return greenLetters, orangeLetters

    def choice(self):
        """Choix d'un mot parmi les mots compatibles
        """
        return random.choice(self.population)

    def approximatePop(self):
        """Etant donné une liste de chaine de caractere (children), 
        les remplacent par les mots du dictionnaire le plus proche.
        """
        newPop = []
        for word in self.children:
            bestWords = []
            bestDist = -1
            for dictWord in self.wordleMindInst.dico:
                if dictWord == word:
                    newPop.append(word)
                    break
                curDist = distance(word, dictWord)
                if curDist >= bestDist:
                    bestDist = curDist
                    bestWords = []
                    bestWords.append(dictWord)
                elif curDist == bestDist:
                    bestWords.append(dictWord)
            newPop.extend(bestWords)
        self.children = np.array(newPop)
        return

    def fitness(self, word):
        """Renvoie la fitness d'un mot : si le mots a un meme nombre de lettre bien placé
        """
        value = 0
        for index in range(len(self.prev_words)):
            green, _ = compareWords(word, self.prev_words[index])
            if green != (self.prev_green[index]):
                value += 1
        return value

    def ajout_Compatible(self):
        """Ajoute les mots compatible depuis les mots generés dans l'ensemble des mots compatibles
        """
        for word in self.children:
            if (word not in self.population.tolist()) and (self.fitness(word) == 0):
                self.population = np.append(self.population,word)
        return 

    def selection(self,n = 0.0):
        """Selection des mots en fonction de la fitness 
        @param n probabilité de base d'etre pris
        @return la liste des mots selectionnés
        """
        to_cross = []
        for ind in self.children:
            fit = self.fitness(ind)
            if (np.random.random()+ fit/len(self.prev_words) - n < 1):
                to_cross.append(ind)

        return to_cross

    def selectBest(self, n =200):
        id = np.argsort([self.fitness(i) for i in self.children])[:n]
        return self.children[id]

    @staticmethod
    def mutateSwap(word):
        """Mutation : change la place de 2 lettres dans une chaine de charactere
        @param word : chaine de charactere
        @return la chaine mutée
        """
        if len(word) == 1:
            return word
        toSwap = np.random.choice(range(len(word)), 2, replace=False)
        out = list(word)
        out[toSwap[1]], out[toSwap[0]] = out[toSwap[0]], out[toSwap[1]]
        return ''.join(out)

    @staticmethod
    def mutateSeq(word):
        """
        Mutation : change la place de 2 sequence de lettre dans une chaine de charactere
        @param word : chaine de charactere
        @return la chaine mutée
        """
        if len(word) == 1:
            return  word
        cut = np.random.choice(range(1, len(word)))
        return word[cut:] + word[:cut]

    @staticmethod
    def mutateLetter(word):
        """
        Mutation : change une lettre par une autre lettre aleatoire dans une chaine de charactere
        @param word : chaine de charactere
        @return la chaine mutée
        """
        toSwap = np.random.choice(range(len(word)))
        newLetter = random.choice(string.ascii_lowercase)
        while newLetter == toSwap:
            newLetter = np.random.choice(string.ascii_lowercase)
        return word[:toSwap] + newLetter + word[toSwap + 1:]

    @staticmethod
    def crossHalf(p1, p2):
        """Croisement : coupe et colle deux mots par leur milieu
        @param p1, p2 : chaine de charactere
        @return out1, out2 : les chaines croisée
        """
        cutOff = np.random.choice(range(1,len(p1)-1))
        out1 = p2[:cutOff] + p1[cutOff:]
        out2 = p1[:cutOff] + p2[cutOff:]
        return out1, out2

    @staticmethod
    def cross(to_cross:list,n=200):
        """Croisement : croise les mots d'indice pair et d'indice impair, si la taille de la liste a retourner n'est pas atteint croise 2 mots au hasard jusqu'a atteindre n
        @param to_cross : liste de mot a croiser
        @param n : taille minimum de la liste a renvoyer
        @return: liste de mots croisés de taille min n
        """
        childrenCross = []
        crossRange = range(1, len(to_cross), 2)
        for i in crossRange:
            childrenCross.extend(Genetics.crossHalf(to_cross[i], to_cross[i-1]))
        return childrenCross

    def generateRandomString(self):
        """Genere une chaine de charactere aleatoire de la taille du mot a trouver
        """
        return ''.join(random.sample(string.ascii_lowercase, self.word_length))

    def initPop(self,n):
        """
        Initialise la liste de mot
        """
        pop = []
        while len(pop) < n:
            pop.extend(self.generateRandomString())
        self.children = np.array(pop)
        self.approximatePop()

    def genetics(self):
        """
        Algorithme genetique pour generer une liste de mot compatible
        """
        curGen = 0
        while curGen < self.maxGen :
            if self.verbose:
                print(f'----- GENERATION {curGen} Pop: {len(self.population)}- Children: {len(self.children)}-----')
            #ajout des mots compatibles
            self.ajout_Compatible()
            #sortie de boucle si maxSize atteint
            if len(self.population)>self.maxSize:
                if self.verbose:
                    print(f'MAX SIZE REACHED')
                break

            #selection des meileurs mots
            to_cross = self.selection()

            #croisement
            childrenCross = self.cross(to_cross)

            #mutation
            childrenMut = []
            for ind in childrenCross:
                childMut = ind
                for mutate in [self.mutateSeq, self.mutateSwap, self.mutateLetter]:
                    if random.random() < self.probMut:
                        childMut = mutate(childMut)
                childrenMut.append(childMut)
            self.children = np.array(childrenCross)
            self.children = np.append(self.children,childrenMut)

            #tranformation des mots mutés par les mots les plus proches du dictionnaire
            self.approximatePop()
            self.children = self.selectBest(200)
            #si generation max atteint, on sort de la boucle
            curGen += 1
            if self.verbose and curGen >= self.maxGen:
                print('Genetic algorithm ran out of generations')
                break
        if self.verbose:
            print(f'------------ DONE WITH GENETICS ------------')
            if (len(self.population)<10):
                print(f'Pop =  {self.population}')
        return 

def main(argv):
    global DICO_INST

    parser = argparse.ArgumentParser(description='Launch app.')
    parser.add_argument('--algorithm','-a', type=str, default="A1",choices=["A1","A2","G"],
                        help='algoritm to choose')
    parser.add_argument('--test', action="store_true",
                        help='test mode')
    parser.add_argument('--verbose','--v', action="store_true",
                        help='verbose')
    args = parser.parse_args()
    verbose = args.verbose    

    if args.test:
        if args.algorithm == "A1":
            a = 1
        elif args.algorithm == "A2":
            a = 2
        elif args.algorithm == "G":
            a = 3
    else:
        while True:
            secret = str(input("\nPlease choose a word \t Type ENTER to choose random word\n"))
            if secret == "":
                while True :
                    try:
                        word_length = int(input("\nSelect a word length\n"))
                        if 4 <= word_length and word_length <= len(max(DICO, key=len)):
                            d = loadDico(DICO_PATH, word_length)
                            secret = np.random.choice(d)
                            break
                        print(f"\nYour world length must be between 4 and{max(DICO, key=len)}")
                    except ValueError:
                        continue
            if secret in DICO:
                print(f"\nThe selected word is {secret}\n")
                word_length = len(secret)
                break
            else:
                print("This word is not in our word database")
        if args.algorithm == "A1":
            a = 4
        elif args.algorithm == "A2":
            a = 5
        elif args.algorithm == "G":
            a = 6

    di = {"word" : [],"wordLength":[],"solved" :[], "time" :[], "tries":[]}
    if a == 1:
        for word_length in range(4,9):
            DICO_INST = loadDico(DICO_PATH, word_length)
            for _ in range(10):
                secret = random.choice(DICO_INST)
                wMind = WordleMind(DICO_INST, secret)

                bChrono = BackTrackChrono(word_length , wMind, verbose)
                bChrono.startGuessing()
                s,ti,tr = bChrono.results()

                di["word"].append(secret)
                di["wordLength"].append(word_length)
                di["solved"].append(s)
                di["time"].append(ti)
                di["tries"].append(tr)
        pd.DataFrame(di).to_csv("./A1.csv")

    if a == 2:
        print("starting backtrack arc")
        for word_length in range(4,9):
            DICO_INST = loadDico(DICO_PATH, word_length)
            for _ in range(10):
                secret = random.choice(DICO_INST)
                wMind = WordleMind(DICO_INST, secret)

                bChronoArc = BackTrackChronoArc(word_length , wMind, verbose)
                bChronoArc.startGuessing()
                s,ti,tr = bChronoArc.results()

                di["word"].append(secret)
                di["wordLength"].append(word_length)
                di["solved"].append(s)
                di["time"].append(ti)
                di["tries"].append(tr)
        pd.DataFrame(di).to_csv("./A2.csv")
       
    if a == 3:
        print("starting genetics")
        for word_length in range(4,9):
            DICO_INST = loadDico(DICO_PATH, word_length)
            for _ in range(10):
                secret = random.choice(DICO_INST)
                wMind = WordleMind(DICO_INST, secret)

                genGuesser = Genetics(word_length, wMind, verbose=verbose)
                genGuesser.startGuessing('')
                s,ti,tr = genGuesser.results()

                di["word"].append(secret)
                di["wordLength"].append(word_length)
                di["solved"].append(s)
                di["time"].append(ti)
                di["tries"].append(tr)
        pd.DataFrame(di).to_csv("./G.csv")
        
    if a == 4:
        DICO_INST = loadDico(DICO_PATH, word_length)
        wMind = WordleMind(DICO_INST, secret)
        bChrono = BackTrackChrono(word_length , wMind, verbose)
        bChrono.startGuessing()
        s,ti,tr = bChrono.results()

    if a == 5:
        DICO_INST = loadDico(DICO_PATH, word_length)
        wMind = WordleMind(DICO_INST, secret)
        bChronoArc = BackTrackChronoArc(word_length , wMind, verbose)
        bChronoArc.startGuessing()
        s,ti,tr = bChronoArc.results()
    
    if a == 6:
        DICO_INST = loadDico(DICO_PATH, word_length)
        wMind = WordleMind(DICO_INST, secret)
        genGuesser = Genetics(word_length, wMind, verbose=verbose)
        genGuesser.startGuessing('')
        s,ti,tr = genGuesser.results()



if __name__ == '__main__':
    main(sys.argv)

