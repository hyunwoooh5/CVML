import numpy as np
import jax
import jax.numpy as jnp
import flax


def jackknife(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(np.delete(x, i, axis=0)*np.delete(w, i, axis=0)) /
            np.mean(np.delete(w, i)) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))*np.sqrt(len(vals)-1)


def bin(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(x[i]*w[i])/np.mean(w[i]) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))/np.sqrt(len(vals)-1)


def bootstrap(xs, ws=None, N=100, Bs=50):
    if Bs > len(xs):
        Bs = len(xs)
    B = len(xs)//Bs
    if ws is None:
        ws = xs*0 + 1
    # Block
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    # Regular bootstrap
    y = x * w
    m = (sum(y) / sum(w))
    ms = []
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real) + 1j*np.std(ms.imag)


# regularizations
def l2_loss(x, alpha):
    return alpha*(x**2).mean()


def l1_loss(x, alpha):
    return alpha*(abs(x)).mean()


def l2_regularization(params):
    # Flatten the nested parameter dict.
    flat_params = flax.traverse_util.flatten_dict(params)
    # Sum up the L2 norm of all parameters where the key ends with 'kernel'
    l2_sum = sum(jnp.sum(param ** 2)
                 for key, param in flat_params.items() if key[-1] == 'kernel')
    return l2_sum


def l1_regularization(params):
    # Flatten the nested parameter dict.
    flat_params = flax.traverse_util.flatten_dict(params)
    # Sum up the L2 norm of all parameters where the key ends with 'kernel'
    l2_sum = sum(jnp.sum(jnp.abs(param))
                 for key, param in flat_params.items() if key[-1] == 'kernel')
    return l2_sum

# For adamW
def decay_mask(params):
    flat = flax.traverse_util.flatten_dict(params)
    mask = {path: (path[-1] == "kernel") for path in flat}
    return flax.traverse_util.unflatten_dict(mask)
