# PTB reading code modified from 
# https://github.com/awni/semantic-rntn/blob/master/tree.py
# credit to Awni Hannun

import collections, cPickle
from nltk.corpus import stopwords
from collections import Counter
import string, sys
UNK = 'UNK'

class Node:
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2 # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open: 
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1])-1) # zero index labels
        node.parent = parent 

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)
        return node

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)

def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]

def loadWordMap():
    import cPickle as pickle
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def words_to_list(node, args):
    sent, revmap = args
    if node.word:
        sent.append(node.word)

def phrase_to_list(node, phrases):
    sent = []
    leftTraverse(node, words_to_list, [sent, None])
    correct = node.label + 1
    phrases.append( (sent, correct) )

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """
    words = collections.defaultdict(int)
    ddir = '../data/sentiment/'

    for file in [ddir + 'train.txt', ddir + 'dev.txt', ddir + 'test.txt']:
        print "Reading trees.."
        with open(file,'r') as fid:
            trees = [Tree(l) for l in fid.readlines()]

        print "Counting words.."
        for tree in trees:
            leftTraverse(tree.root,nodeFn=countWords,args=words)
        
        print len(words)

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    with open('../data/sentiment/wordMapAll.bin','w') as fid:
        cPickle.dump(wordMap,fid)

    return wordMap

def loadTrees(dataSet='train', wmap=loadWordMap):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = wmap
    file = '../data/sentiment/%s.txt'%dataSet
    print "Reading trees.."
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees

def preprocess(sents, wmap, binary=False):

    proc = []
    for sent, label in sents:
        proc.append([sent, label])

    return proc

def process_trees(wmap):
    for split in ['train', 'dev', 'test']:
        trees = loadTrees(dataSet=split, wmap=wmap)
        print len(trees)
        cPickle.dump(trees, open('../data/sentiment/' + split + '_alltrees', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
      

def acquire_all_phrases(tree, phrases):
    leftTraverse(tree.root,nodeFn=phrase_to_list,args=phrases)
    return phrases

if __name__=='__main__':

    wmap = buildWordMap()
    print 'num words: ', len(wmap)
    process_trees(wmap)
    train = cPickle.load(open('../data/sentiment/train_alltrees', 'rb'))
    dev = cPickle.load(open('../data/sentiment/dev_alltrees', 'rb'))
    test = cPickle.load(open('../data/sentiment/test_alltrees', 'rb'))

    revMap = {}
    for k, v in wmap.iteritems():
        revMap[v] = k
    print len(train), len(dev), len(test)
    
    # store train root labels
    t_sents = []
    for tree in train:
        sent = []
        leftTraverse(tree.root,nodeFn=words_to_list,args=[sent,revMap])
        t_sents.append([sent, tree.root.label + 1])

    print [revMap[x] for x in t_sents[0][0]]
    print 'num train instances ', len(t_sents)
    c = Counter()
    for sent, label in t_sents:
        c[label] += 1
    print c
    cPickle.dump(t_sents, open('../data/sentiment/train-rootfine', 'wb'))

    # store both phrases and roots for dev / test
    dev_sents = []
    for tree in dev:
        sent = []
        leftTraverse(tree.root,nodeFn=words_to_list,args=[sent,revMap])
        dev_sents.append([sent, tree.root.label + 1])

    print [revMap[x] for x in dev_sents[0][0]]
    print 'dev phrase length ', len(dev_sents)
    c = Counter()
    for sent, label in dev_sents:
        c[label] += 1
    print c
    cPickle.dump(dev_sents, open('../data/sentiment/dev-rootfine', 'wb'))

    test_sents = []
    for tree in test:
        sent = []
        leftTraverse(tree.root,nodeFn=words_to_list,args=[sent,revMap])
        test_sents.append([sent, tree.root.label + 1])
                 
    print [revMap[x] for x in test_sents[0][0]]
    print 'test phrase length ', len(test_sents)
    c = Counter()
    for sent, label in test_sents:
        c[label] += 1
    print c
    cPickle.dump(test_sents, open('../data/sentiment/test-rootfine', 'wb'))