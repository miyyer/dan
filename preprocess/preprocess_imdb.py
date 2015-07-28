from glob import glob
import cPickle
import random

def compute_vocab():
    vocab = []
    vdict = {}
    trneg = glob('../data/aclimdb/train/neg/*.txt')
    trpos = glob('../data/aclimdb/train/pos/*.txt')
    tneg = glob('../data/aclimdb/test/neg/*.txt')
    tpos = glob('../data/aclimdb/test/pos/*.txt')

    split = []
    for fold in [trneg, trpos, tneg, tpos]:
        fold_docs = []
        for fname in fold:
            doc = []
            f = open(fname, 'r')
            for line in f:
                line = line.strip().replace('.', '').replace(',', '')
                line = line.replace(';', '').replace('<br />', ' ')
                line = line.replace(':', '').replace('"', '')
                line = line.replace('(', '').replace(')', '')
                line = line.replace('!', '').replace('*', '')
                line = line.replace(' - ', ' ').replace(' -- ', '')
                line = line.replace('?', '')
                line = line.lower().split()

                for word in line:
                    try:
                        vdict[word]
                    except:
                        vocab.append(word)
                        vdict[word] = len(vocab) - 1

                    doc.append(vdict[word])

            fold_docs.append(doc)
        split.append(fold_docs)


    train = []
    test = []
    for i in range(0, len(split)):
        for doc in split[i]:
            if i == 0:
                train.append((doc, 0))
            elif i == 1:
                train.append((doc, 1))
            elif i == 2:
                test.append((doc, 0))
            elif i == 3:
                test.append((doc, 1))

    print len(train), len(test)

    random.shuffle(train)
    random.shuffle(test)

    for x in range(3000, 3020):
        print i, train[x][1], ' '.join(vocab[x] for x in train[x][0])
        print '\n'

    cPickle.dump([train, test, vocab, vdict], open('../data/aclimdb/imdb_splits', 'wb'),\
        protocol=cPickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    compute_vocab()