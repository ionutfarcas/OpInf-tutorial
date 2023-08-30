import numpy as np

def get_x_sq(X):
    if len(np.shape(X))==1: # if X is a vector
        r = np.size(X)
        prods = []
        for i in range(r):
            temp = X[i]*X[i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods))

    elif len(np.shape(X))==2: # if X is a matrix
        K,r = np.shape(X)
        
        prods = []
        for i in range(r):
            temp = np.transpose(np.broadcast_to(X[:,i],(r-i,K)))*X[:,i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods),axis=1)

    else:
        print('invalid input size for helpers.get_x_sq')
    return X2

def get_err(A, B, opt='max-rel-2'):
    if opt == 'max-rel-2':
        error = np.max(np.sqrt(np.sum( (B-A)**2,axis=1) / np.sum(A**2,axis=1)))
    
    elif opt == 'er1':
        # entry-wise relative error
        error = np.linalg.sum(np.abs( (B-A)/A))/np.size(A)
    
    elif opt == 'mean-rel-2':
        error = np.mean(np.sqrt(np.sum((B-A)**2,axis=1) / np.sum(A**2,axis=1)))
    
    return error

def evolve_opinf_difference_model(s0, n_steps, f, verbose=False):

    s       = np.zeros((np.size(s0), n_steps))
    is_nan  = False

    s[:, 0] = s0
    for i in range(n_steps - 1):

        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):

            if verbose:
                print('NaN encountered at iteration '+str(i + 1))

            is_nan = True
            break

    return is_nan, s