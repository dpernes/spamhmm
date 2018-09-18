from sklearn.base import BaseEstimator
from spamhmm import SpaMHMM
from hmmlearn.utils import iter_from_X_lengths
import numpy as np


class KHMM(BaseEstimator):
    def __init__(self, n_nodes, n_components, n_features, emission='gaussian',
                 n_iter=10, tol=1e-2, verbose=False, name='khmm'):
        super(KHMM, self).__init__()

        self.n_nodes = n_nodes
        self.n_components = n_components
        self.n_features = n_features
        self.emission = emission
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.name = name

    def init_params(self):
        self.hmm = [SpaMHMM(n_nodes=1,
                            mix_dim=1,
                            n_components=self.n_components,
                            n_features=self.n_features,
                            emission=self.emission,
                            graph=None,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            verbose=self.verbose,
                            name=(self.name+'_'+str(i)
                                  for i in range(self.n_nodes)))]

    def fit(self, X, y, lengths=None, valid_data=None):
        trainloss_hist = []

        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)
        elif lengths is None:
            lengths = [X.shape[0]]

        if valid_data is not None:
            X_valid, y_valid, lengths_valid = valid_data

            if type(X_valid) == list:
                lengths_valid = [x.shape[0] for x in X_valid]
                X_valid = np.concatenate(X_valid)
                y_valid = np.array(y_valid)

            validloss_hist = []

        self.init_params()

        for k in range(self.n_nodes):
            lengthsk = [lengths[i] for i in range(len(lengths)) if y[i] == k]
            if not lengthsk:
                continue
            Xk = np.concatenate([X[i:j, :] for seq_idx, (i, j)
                                 in enumerate(iter_from_X_lengths(X, lengths))
                                 if y[seq_idx] == k], axis=0)
            yk = np.array([0 for i in range(len(lengthsk))])

            if valid_data is not None:
                Xk_valid = np.concatenate([X_valid[i:j, :] for seq_idx, (i, j)
                                           in enumerate(iter_from_X_lengths(
                                                            X_valid,
                                                            lengths_valid))
                                           if y_valid[seq_idx] == k], axis=0)
                lengthsk_valid = [lengths_valid[i]
                                  for i in range(len(lengths_valid))
                                  if y_valid[i] == k]
                yk_valid = np.array([0 for i in range(len(lengthsk_valid))])

                trainlossk, validlossk = (self.hmm[k]
                                          .fit(Xk, yk, lengths=lengthsk,
                                               valid_data=(Xk_valid,
                                                           yk_valid,
                                                           lengthsk_valid)))
                validloss_hist.append(validlossk)
            else:
                trainlossk = self.hmm[k].fit(Xk, yk, lengths=lengthsk)

            trainloss_hist.append(trainlossk)

        if valid_data is not None:
            return trainloss_hist, validloss_hist
        else:
            return trainloss_hist

    def score(self, X, y, lengths=None):
        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)
        elif lengths is None:
            lengths = [X.shape[0]]

        Nseqs = y.shape[0]
        score = 0
        for k in range(self.n_nodes):
            lengthsk = [lengths[i] for i in range(len(lengths)) if y[i] == k]
            if not lengthsk:
                continue
            Nseqsk = len(lengthsk)
            Xk = np.concatenate([X[i:j, :]
                                 for seq_idx, (i, j) in
                                 enumerate(iter_from_X_lengths(X, lengths))
                                 if y[seq_idx] == k], axis=0)
            yk = np.array([0 for i in range(Nseqsk)])

            scorek = self.hmm[k].score(Xk, yk, lengths=lengthsk)
            score += scorek * Nseqsk
        score /= Nseqs

        return score

    def scores_per_seq(self, X, y, lengths=None):
        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)
        elif lengths is None:
            lengths = [X.shape[0]]

        N = y.shape[0]
        log_likelihood = np.zeros(N)
        for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
            log_likelihood[seq_idx] = (self.hmm[y[seq_idx]]
                                       .scores_per_seq(X[i:j, :],
                                                       np.array([0])))

        return log_likelihood

    def sample(self, y, n_samples, Xpref=None):
        X, _, state_seq = self.hmm[y].sample(0, n_samples, Xpref)

        return X, state_seq
