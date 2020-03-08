from scipy.linalg import svd
import numpy as np


def GetPCAComponents(diagonal, epsilon=0.5):
    sum_diagonal = np.sum(diagonal)
    cumsum_diagonal = np.cumsum(diagonal)
    result = cumsum_diagonal/sum_diagonal
    Nr = result[result < epsilon].shape[0]
    return np.diag(diagonal[0:Nr+1])


def GetPCAComponentsNr(diagonal, Nc=70):
    return np.diag(diagonal[0:Nc])


def ComputePCA(data, epsilon=0.4, Nc=70, computeAll=True):
    data_vector = data.reshape(-1, np.prod(data.shape[-3:]))
    mean_ = np.mean(data_vector, axis=0)
    data_c = data_vector-mean_
    const = 1/np.sqrt(data.shape[0]-1)

    U, s, V = svd(data_c.T, full_matrices=False)
    if computeAll:
        S = GetPCAComponents(s, epsilon=epsilon)
    else:
        S = GetPCAComponentsNr(s, Nc=Nc)
    nz = S.shape[0]

    U_ = U[:, :nz]
    Us = np.dot(U_, S)

    Si = np.diag(s[:nz]**(-1))
    Us_inv = np.dot(Si, U_.T)

    if computeAll:
        Us = np.dot(Us, U_.T)
        Us_inv = np.dot(U_, Us_inv)
        nz = data.shape[-1]*data.shape[-2]*data.shape[-3]
    return Us, Us_inv, const, nz, mean_, data_c
