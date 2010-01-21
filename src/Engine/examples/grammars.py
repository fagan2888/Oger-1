
from numpy import random

if __name__ == '__main__':
    print 'This module is meant to be imported by grammar_task.py'

class elman_grammar:
    """This class defines a probabilistic context free grammar that is similar
    (if not identical) to the grammar used in Elman (1991).  The grammar
    contains a couple of rules that can be called individually to give parts of
    sentences.  To generate a sentence one uses the 'S()' method.
    """
    
    def __init__(self, coding='string'):
        self.set_coding(coding)
        
    def set_coding(self, coding):
        """Change the way the words are represented.
         
        Setting coding to 'string' results in the use of character strings.
        Any other value leads to the use of integer coding.
        """
        
        if coding == 'string':
            self.lexicon = ['boy', 'girl', 'cat', 'dog', 'boys', 'girls', 'cats', 'dogs',\
                            'chase', 'feed', 'see', 'hear', 'walk', 'live', 'chases',\
                                'feeds', 'sees', 'hears', 'walks', 'lives', 'John', 'Mary', 'who', '.']
            self.coding = 'string'
        else:
            self.lexicon = range(22)
            self.coding = 'numerical'
        
        l = self.lexicon
        self.nouns = l[:8]
        self.nouns_s = l[:4]
        self.nouns_p = l[4:8]
        self.verbs = l[8:20]
        self.verbs_p = l[8:14]
        self.verbs_s = l[14:20]
        self.verbs_t = l[8:10] + l[14:16]
        self.verbs_o = l[10:12] + l[16:18]
        self.verbs_u = l[12:14] + l[18:20]
        self.PropNs = l[20:22]
        self.END = l[-1]
        self.WHO = l[-2]
    
    def S(self):
        singular = random.random_integers(1, 2)
        return self.NP(singular) + self.VP(singular) + [self.END]
    
    def RC(self, singular=0):
        NP = self.NP
        V = self.V
        VP = self.VP
        singular2 = random.random_integers(1, 2)
        transitive = random.random_integers(0, 2)
        if transitive != 2:
            choice = random.random_integers(0, 1)
            if choice == 0:
                return [self.WHO] + NP(singular2) + V(singular2, transitive)
            elif choice == 1:
                return [self.WHO] + V(singular, transitive) + NP()
        else:
            return [self.WHO] + V(singular, transitive)
    
    def NP(self, singular=0):
        if singular == 0:
            singular = random.random_integers(1, 2)
        if singular == 1:
            f = 3 * [[]]
            f[0] = lambda: self.N(singular)
            f[1] = lambda: self.N(singular) + self.RC(singular)
            f[2] = lambda: self.PropN()
            return f[random.random_integers(0, 2)]()
        else:
            f = 2 * [[]]
            f[0] = lambda: self.N(singular)
            f[1] = lambda: self.N(singular) + self.RC(singular)
            return f[random.random_integers(0, 1)]()
    
    def PropN(self):
        return [self.PropNs[random.random_integers(0, 1)]]
    
    def N(self, singular=0):
        if singular == 1:
            return [self.nouns_s[random.random_integers(0, len(self.nouns_s) - 1)]]
        elif singular == 2:
            return [self.nouns_p[random.random_integers(0, len(self.nouns_p) - 1)]]
        else:
            return [self.nouns[random.random_integers(0, len(self.nouns) - 1)]]
    
    
    def V(self, singular=0, transitive=0):
        if singular == 1:
            pool1 = set(self.verbs_s)
        elif singular == 2:
            pool1 = set(self.verbs_p)
        else:
            pool1 = set(self.verbs)
        if transitive == 1:
            pool2 = set(self.verbs_t)
        elif transitive == 2:
            pool2 = set(self.verbs_u)
        else:
            pool2 = set(self.verbs_o)
        verbs = list(pool1.intersection(pool2))
        return [verbs[random.random_integers(0, len(verbs) - 1)]]
    
    def VP(self, singular=0):
        if singular == 0:
            singular = random.random_integers(1, 2)
        transitive = random.random_integers(0, 2)
        if transitive == 0:
            if random.sample() > .5:
                return self.V(singular, transitive)
            else:
                return self.V(singular, transitive) + self.NP()
        elif transitive == 1:  # Transitive verb.
            return self.V(singular, transitive) + self.NP()
        else:
            return self.V(singular, transitive)


class simple_pcfg:
    """A very simple grammar that allows for variable vocabulary sizes.
    """
    
    def __init__(self, nouns=None, verbs=None, coding='string'):
        self.nouns = nouns
        self.verbs = verbs
        self.type = type
        if coding == 'string':
            if self.nouns == None:
                self.nouns = ['hamsters', 'boys', 'girls', 'pigs', 'computers', 'clowns', 'engineers']
            if self.verbs == None:
                self.verbs = ['follow', 'simulate', 'eat', 'collect', 'love', 'help']
            self.THAT = 'that'
            self.END = '.'
            self.WITH = 'with'
            self.FROM = 'from'
            self.THE = 'the'
        else:
            self.THAT = 0
            self.END = 1
            self.WITH = 2
            self.FROM = 3
            self.nouns = range(4, 4 + len(nouns))
            self.verbs = range(4 + len(nouns), 4 + len(nouns) + len(verbs))
    
    def S(self):
        return self.NP() + self.V() + self.NP() + [self.END]
    
    def SRC(self):
        NP = self.NP
        V = self.V
        return [self.THAT] + V() + NP()
    
    def ORC(self):
        N = self.N
        V = self.V
        return [self.THAT] + N() + V()
    
    def NP(self):
        f = 4 * [[]]
        f[0] = lambda: self.N()
        f[1] = lambda: self.N() + self.SRC()
        f[2] = lambda: self.N() + self.ORC()
        f[3] = lambda: self.N() + self.PP()
        return f[random.random_integers(0, 3)]()
    
    def PP(self):
        if random.sample() > .5:
            return [self.FROM] + self.N()
        else:
            return [self.WITH] + self.N()
    
    def N(self):
        if random.sample() > .5:
            return [self.nouns[random.random_integers(0, len(self.nouns) - 1)]]
        else:
            return [self.THE] + [self.nouns[random.random_integers(0, len(self.nouns) - 1)]]
    
    def V(self):
        return [self.verbs[random.random_integers(0, len(self.verbs) - 1)]]



