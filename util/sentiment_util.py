from numpy import *
import cPickle

def unroll_params(arr, d, dh, len_voc, deep=1, labels=2, wv=True):

    mat_size = dh * dh
    ind = 0

    params = []
    if deep > 0:
        params.append(arr[ind : ind + d * dh].reshape( (dh, d) ))
        ind += d * dh
        params.append(arr[ind : ind + dh].reshape( (dh, ) ))
        ind += dh

        for i in range(1, deep):
            params.append(arr[ind : ind + mat_size].reshape( (dh, dh) ))
            ind += mat_size
            params.append(arr[ind : ind + dh].reshape( (dh, ) ))
            ind += dh

    params.append(arr[ind: ind + labels * dh].reshape( (labels, dh)))
    ind += dh * labels
    params.append(arr[ind: ind + labels].reshape( (labels, )))
    ind += labels
    if wv:
        params.append(arr[ind : ind + len_voc * d].reshape( (d, len_voc)))
    return params

# roll all parameters into a single vector
def roll_params(params):
    return concatenate( [p.ravel() for p in params])


# initialize all parameters to magic
def init_params(d, dh, deep=1, labels=2):
    # magic_number = 2. / d
    magic_number = 0.08
    params = []
    if deep > 0:
        params.append( (random.rand(dh, d) * 2 - 1) * magic_number)
        params.append( (random.rand(dh, ) * 2 - 1) * magic_number)

        for i in range(1, deep):
            params.append( (random.rand(dh, dh) * 2 - 1) * magic_number)
            params.append( (random.rand(dh, ) * 2 - 1) * magic_number)

    params.append((random.rand(labels, dh) * 2 - 1) * magic_number)
    params.append((random.rand(labels, ) * 2 - 1) * magic_number)
    return params

# returns list of zero gradients which backprop modifies
def init_grads(d, dh, len_voc, deep=1, labels=2, wv=True):

    grads = []
    if deep > 0:
        grads.append(zeros((dh, d)))
        grads.append(zeros(dh, ))

        for i in range(1, deep):
            grads.append(zeros( (dh, dh) ))
            grads.append(zeros( (dh, ) ))

    grads.append(zeros( (labels, dh) ))
    grads.append(zeros( (labels, ) ))
    if wv:
        grads.append(zeros((d, len_voc)))
    return grads

# random embedding matrix for gradient checks
def gen_rand_we(len_voc, d):
    r = sqrt(6) / sqrt(257)
    we = random.rand(d, len_voc) * 2 * r - r
    return we