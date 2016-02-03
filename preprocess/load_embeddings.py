from numpy import *
import cPickle, zipfile

vec_file = zipfile.ZipFile('../data/glove.840B.300d.zip', 'r').open('glove.840B.300d.txt', 'r')
all_vocab = {}
print 'loading vocab...'
wmap = cPickle.load(open('../data/sentiment/wordMapAll.bin', 'rb'))
revMap = {}
for word in wmap:
    revMap[wmap[word]] = word

for line in vec_file:
    split = line.split()
    try:
        x = wmap[split[0]]
        all_vocab[split[0]] = array(split[1:])
        all_vocab[split[0]] = all_vocab[split[0]].astype(float)
    except:
        pass

print len(wmap), len(all_vocab)
d = len(all_vocab['the'])

We = empty( (d, len(wmap)) )

print 'creating We for ', len(wmap), ' words'
unknown = []

for i in range(0, len(wmap)):
    word = revMap[i]
    try:
        We[:, i] = all_vocab[word]
    except KeyError:
        unknown.append(word)
        print 'unknown: ', word
        We[:, i] = all_vocab['unknown']

print 'num unknowns: ', len(unknown)
print We.shape

print 'dumping...'
cPickle.dump( We, open('../data/sentiment_We', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
