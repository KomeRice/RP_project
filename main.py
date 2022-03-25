import sys
from collections import defaultdict
from constraint import *

DICO_PATH = './dico.txt'

def loadDico(path, n):
    with open(path,'r') as f:
        return [line.rstrip('\n') for line in f if len(line.rstrip('\n')) == n]

def checkWord(word,secret):
    # Right letter right place
    greenLetters = 0
    # Right letter wrong place
    orangeLetters = 0

    remainWord = defaultdict(lambda: 0)
    remainSecret = defaultdict(lambda: 0)

    for i in range(len(word)):
        if word[i] == secret[i]:
            greenLetters += 1
        else:
            remainWord[word[i]] += 1
            remainSecret[secret[i]] += 1

    for i in remainWord:
        if remainSecret[i] != 0:
            orangeLetters += min(remainWord[i], remainSecret[i])

    return greenLetters, orangeLetters



def main(argv):
    word_length = -1
    try:
        word_length = int(argv[1])
    except IndexError:
        print(f'Not enough arguments given, expected 2 got {len(argv)}\nFormat: python3 ./main.py <word_length>')
    except ValueError:
        print('Invalid word length given as parameter')
    dico = loadDico(DICO_PATH, word_length)
    print(len(dico))
    print(dico)

if __name__ == '__main__':
    main(sys.argv)

