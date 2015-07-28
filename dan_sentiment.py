from numpy import *
from util.sentiment_util import *
from util.math_util import *
from util.adagrad import Adagrad
import cPickle, time, argparse
from collections import Counter

# compute model accuracy on a given fold
def validate(data, fold, params, deep, f=relu):

    correct = 0.
    total = 0.

    for sent, label in data:

        if len(sent) == 0:
            continue

        av = average(params[-1][:, sent], axis=1)

        # forward prop
        acts = zeros((deep, dh))  
        for i in range(0, deep):
            start = i * 2
            prev = av if i == 0 else acts[i - 1]
            acts[i] = f(params[start].dot(prev) + params[start + 1])

        Ws = params[deep * 2]
        bs = params[deep * 2 + 1]
        if deep == 0:
            pred = softmax(Ws.dot(av) + bs).ravel()

        else:
            pred = softmax(Ws.dot(acts[-1]) + bs).ravel()

        if argmax(pred) == label:
            correct += 1

        total += 1

    print 'accuracy on ', fold, correct, total, str(correct / total), '\n'
    return correct / total

# does both forward and backprop
def objective_and_grad(data, params, d, dh, len_voc, deep, labels, f=relu, df=drelu, compute_grad=True, word_drop=0.3, rho=1e-4, fine_tune=True):

    params = unroll_params(params, d, dh, len_voc, deep=deep, labels=labels)
    grads = init_grads(d, dh, len_voc, deep=deep, labels=labels)
    error_sum = 0.0

    for sent,label in data:

        if len(sent) == 0:
            continue

        # store each layer's normalized and unnormalized acts
        acts = zeros((deep, dh))

        target = zeros(labels)
        target[label] = 1.0

        # input is average of all nouns in sentence
        curr_sent = []
        mask = random.rand(len(sent)) > word_drop
        for index, keep in enumerate(mask):
            if keep:
                curr_sent.append(sent[index])

        # all examples must have at least one word
        if len(curr_sent) == 0:
            curr_sent = sent

        av = average(params[-1][:, curr_sent], axis=1)

        # forward prop
        for i in range(0, deep):
            start = i * 2
            prev = av if i == 0 else acts[i - 1]
            acts[i] = f(params[start].dot(prev) + params[start + 1])

        # compute softmax error
        Ws = params[deep * 2]
        bs = params[deep * 2 + 1]

        if deep == 0:
            pred = softmax(Ws.dot(av) + bs).ravel()
            error_sum += crossent(target, pred)
            soft_delta = dcrossent(target, pred)
            grads[deep * 2] += outer(soft_delta, av)
            grads[deep * 2 + 1] += soft_delta
            delta = Ws.T.dot(soft_delta)
            if fine_tune:
                grads[-1][:, curr_sent] += delta.reshape((d, 1)) / len(curr_sent)

        else:
            pred = softmax(Ws.dot(acts[-1]) + bs).ravel()
            error_sum += crossent(target, pred)
            soft_delta = dcrossent(target, pred)
            grads[deep * 2] += outer(soft_delta, acts[-1])
            grads[deep * 2 + 1] += soft_delta

            # backprop
            prev_delta = Ws.T.dot(soft_delta)
            for i in range(deep - 1, -1, -1):
                start = i * 2
                deriv = df(acts[i])
                delta = deriv * prev_delta

                if i > 0:
                    grads[start] += outer(delta, acts[i-1])
                    grads[start + 1] += delta
                    prev_delta = params[start].T.dot(delta)

                else:
                    grads[0] += outer(delta, av)
                    grads[1] += delta

                    if fine_tune:
                        grads[-1][:, curr_sent] += params[0].T.dot(delta).reshape((d, 1)) / len(curr_sent)

    for index in range(0, len(params)):
        error_sum += 0.5 * rho * sum(params[index] ** 2)
        grads[index] += rho * params[index]

    cost = error_sum / len(data)
    grad = roll_params(grads) / len(data)

    if compute_grad:
        return cost, grad
    else:
        return cost


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='sentiment DAN')
    parser.add_argument('-data', help='location of dataset', default='data/sentiment/')
    parser.add_argument('-vocab', help='location of vocab', default='data/sentiment/wordMapAll.bin')
    parser.add_argument('-We', help='location of word embeddings', default='data/sentiment_all_We')
    parser.add_argument('-rand_We', help='randomly init word embeddings', type=int, default=0)
    parser.add_argument('-binarize', help='binarize labels', type=int, default=0)
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-dh', help='hidden dimension', type=int, default=300)
    parser.add_argument('-deep', help='number of layers', type=int, default=3)
    parser.add_argument('-drop', help='dropout probability', type=float, default=0.3)
    parser.add_argument('-rho', help='regularization weight', type=float, default=1e-4)
    parser.add_argument('-labels', help='number of labels', type=int, default=5)
    parser.add_argument('-ft', help='fine tune word vectors', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, x for history and y for lit', type=int,\
                        default=15)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=5)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=50)
    parser.add_argument('-lr', help='adagrad initial learning rate', type=float, default=0.005)
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='models/sentiment_params.pkl')

    args = vars(parser.parse_args())
    d = args['d']
    dh = args['dh']

    # load data
    train = cPickle.load(open(args['data']+'train-rootfine', 'rb'))
    dev = cPickle.load(open(args['data']+'dev-rootfine', 'rb'))
    test = cPickle.load(open(args['data']+'test-rootfine', 'rb'))
    vocab = cPickle.load(open(args['vocab'], 'rb'))
    len_voc = len(vocab)

    for split in [train, dev, test]:
        c = Counter()
        tot = 0
        for sent, label in split:
            c[label] += 1
            tot += 1
        print c, tot

    if args['rand_We']:
        print 'randomly initializing word embeddings...'
        orig_We = (random.rand(d, len_voc) * 2 - 1) * 0.08
    else:
        print 'loading pretrained word embeddings...'
        orig_We = cPickle.load(open(args['We'], 'rb'))

    # output log and parameter file destinations
    param_file = args['output']
    log_file = param_file.split('_')[0] + '_log'

    # generate params / We
    params = init_params(d, dh, deep=args['deep'], labels=args['labels'])

    # add We matrix to params
    params += (orig_We, )
    r = roll_params(params)

    dim = r.shape[0]
    print 'parameter vector dimensionality:', dim

    log = open(log_file, 'w')

    # minibatch adagrad training
    ag = Adagrad(r.shape, args['lr'])
    min_error = float('inf')

    for epoch in range(0, args['num_epochs']):

        lstring = ''

        # create mini-batches
        random.shuffle(train)
        batches = [train[x : x + args['batch_size']] for x in xrange(0, len(train), 
                   args['batch_size'])]

        epoch_error = 0.0
        ep_t = time.time()
        for batch_ind, batch in enumerate(batches):
            now = time.time()
            err, grad = objective_and_grad(batch, r, d, dh, len_voc, 
                args['deep'], args['labels'], word_drop=args['drop'], 
                fine_tune=args['ft'], rho=args['rho'])

            update = ag.rescale_update(grad)
            r = r - update
            lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                    ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
            log.write(lstring + '\n')
            log.flush()
            epoch_error += err

        # done with epoch
        print time.time() - ep_t
        print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
        lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                 + ' min error = ' + str(min_error) + '\n'
        log.write(lstring)
        log.flush()

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            params = unroll_params(r, d, dh, len_voc, deep = args['deep'], labels=args['labels'])
            # d_score = validate(dev, 'dev', params, args['deep'])
            cPickle.dump( params, open(param_file, 'wb'))

        log.flush()

        # reset adagrad weights
        if epoch % args['adagrad_reset'] == 0 and epoch != 0:
            ag.reset_weights()

    log.close()

    # compute test score
    params = unroll_params(r, d, dh, len_voc, deep = args['deep'], labels=args['labels'])
    t_score = validate(test, 'test', params, args['deep'])