from numpy import *
import _pickle as cPickle, zipfile

# vec_file = zipfile.ZipFile('../data/glove.840B.300d.zip', 'r').open('glove.840B.300d.txt', 'r')
vec_file = open('../data/glove.840B.300d.txt', 'r')
print('loading vocab...')
wmap = cPickle.load(open('../data/sentiment/wordMapAll.bin', 'rb'))
revMap = {}
for word in wmap:
    revMap[wmap[word]] = word
all_vocab = {}
for line in vec_file:
    split = line.split()
    try:
        x = wmap[split[0]]
        all_vocab[split[0]] = array(split[1:])
        all_vocab[split[0]] = all_vocab[split[0]].astype(float)
    except:
        pass

print(len(wmap), len(all_vocab))
d = len(all_vocab['the'])

We = empty( (d, len(wmap)))

print('creating We for ', len(wmap), ' words')
unknown = []
wrong_shape = []

for i in range(0, len(wmap)):
    word = revMap[i]
    try:
        We[:, i] = all_vocab[word]
    except KeyError:
        unknown.append(word)
        print('unknown: ', word)
        We[:, i] = all_vocab['unknown']
    except ValueError:
        print('value error', word, all_vocab[word][:3], all_vocab[word].shape)
        wrong_shape.append(word)
        start = len(all_vocab[word]) - d
        We[:, i] = all_vocab[word][start:]

print('num unknowns: ', len(unknown))
print('num wrong shapes', len(wrong_shape))
print(We.shape)

print('dumping...')
cPickle.dump( We, open('../data/sentiment_We', 'wb'))
