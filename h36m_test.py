import numpy as np
from spamhmm import SpaMHMM
from khmm import KHMM
from hmmlearn import hmm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from hmmlearn.utils import iter_from_X_lengths
import argparse
import pickle
from utils import h36m_data_utils as data_utils
from utils.global_utils import np2lst, lst2np, get_S_1HMM, get_S_KHMM
import utils.h36m_processdata as poseDataset

parser = argparse.ArgumentParser(description='Script configurations.')
parser.add_argument('--action', type=str, default='walking')
parser.add_argument('--data_path', type=str, default='./h36m_data')
parser.add_argument('--models_path', type=str, default='./models')
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--train_mhmm', type=int, default=1)
parser.add_argument('--train_spamhmm', type=int, default=1)
parser.add_argument('--train_1hmm', type=int, default=1)
parser.add_argument('--train_khmm', type=int, default=1)
parser.add_argument('--train_verbose', type=int, default=0)
parser.add_argument('--mix_dim', type=int, default=18)
parser.add_argument('--n_components', type=int, default=12)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--reg', type=float, default=0.05)
parser.add_argument('--n_iter_mstep', type=int, default=100)
parser.add_argument('--lr_mstep', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sequence_length', type=int, default=150)
parser.add_argument('--sequence_overlap', type=int, default=50)
parser.add_argument('--forecast_prefix', type=int, default=50)
parser.add_argument('--forecast_suffix', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=100)
args = parser.parse_args()

np.random.seed(args.seed)

(data_train, data_test, data_stats,
 Xforecast_pref, Xforecast_suff,
 pruned_joints, padded_dims) = poseDataset.runall(args.data_path, args.action,
                                                  args.sequence_length,
                                                  args.sequence_overlap,
                                                  args.forecast_prefix,
                                                  args.forecast_suffix)

A = data_utils.prune_adj_matrix(data_utils.build_adj_matrix(), pruned_joints)

Xtrain, ytrain, lengths_train = data_utils.build_inputs(data_train)
Xtest, ytest, lengths_test = data_utils.build_inputs(data_test)

K = len(np.unique(ytrain))
D = Xtrain.shape[1]
M = 1
S = 1

Xtrain, ytrain = np2lst(Xtrain, ytrain, lengths_train)
Xtest, ytest = np2lst(Xtest, ytest, lengths_test)

mhmm_path = args.models_path + '/mhmm_h36m_' + args.action + '.pkl'
if args.train and args.train_mhmm:
    if args.cv:
        print('Training MHMM using CV...')
        mhmm = SpaMHMM(n_nodes=K,
                       mix_dim=M,
                       n_components=S,
                       n_features=D,
                       graph=None,
                       n_iter=args.n_iter,
                       verbose=args.train_verbose,
                       name='mhmm')
        pgrid = {'mix_dim': [10, 14, 18], 'n_components': [5, 8, 12]}

        mhmmCV = GridSearchCV(mhmm, pgrid, cv=StratifiedKFold(3, shuffle=True),
                              n_jobs=-1)
        mhmmCV.fit(Xtrain, ytrain)
        print('Best parameters')
        print(mhmmCV.best_params_)
        f = open(mhmm_path, 'wb')
        pickle.dump(mhmmCV.best_estimator_, f)
        mhmm = mhmmCV.best_estimator_
        M = mhmm.mix_dim
        S = mhmm.n_components
    else:
        print('Training MHMM...')
        M = args.mix_dim
        S = args.n_components
        mhmm = SpaMHMM(n_nodes=K,
                       mix_dim=M,
                       n_components=S,
                       n_features=D,
                       graph=None,
                       n_iter=args.n_iter,
                       verbose=args.train_verbose,
                       name='mhmm')
        mhmm.fit(Xtrain, ytrain)
        f = open(mhmm_path, 'wb')
        pickle.dump(mhmm, f)
else:
    f = open(mhmm_path, 'rb')
    mhmm = pickle.load(f)
    M = mhmm.mix_dim
    S = mhmm.n_components

print('MHMM trained/loaded!')

spamhmm_path = args.models_path + '/spamhmm_h36m_' + args.action + '.pkl'
if args.train and args.train_spamhmm:
    if args.cv:
        print('Training SpaMHMM using CV...')
        spamhmm = SpaMHMM(n_nodes=K,
                          mix_dim=M,
                          n_components=S,
                          n_features=D,
                          n_iter=args.n_iter,
                          n_iter_mstep=args.n_iter_mstep,
                          lr_mstep=args.lr_mstep,
                          verbose=args.train_verbose,
                          name='spamhmm')
        pgrid = {'graph': [1e-4*A, 5e-4*A, 1e-3*A, 5e-2*A, 1e-1*A, 5e-1*A]}

        spamhmmCV = GridSearchCV(spamhmm, pgrid,
                                 cv=StratifiedKFold(3, shuffle=True),
                                 n_jobs=-1)
        spamhmmCV.fit(Xtrain, ytrain)
        print('Best parameters')
        print(spamhmmCV.best_params_)
        f = open(spamhmm_path, 'wb')
        pickle.dump(spamhmmCV.best_estimator_, f)
        spamhmm = spamhmmCV.best_estimator_
    else:
        print('Training SpaMHMM...')
        G = args.reg*A
        spamhmm = SpaMHMM(n_nodes=K,
                          mix_dim=M,
                          n_components=S,
                          n_features=D,
                          graph=G,
                          n_iter=args.n_iter,
                          n_iter_mstep=args.n_iter_mstep,
                          lr_mstep=args.lr_mstep,
                          verbose=args.train_verbose,
                          name='spamhmm')
        spamhmm.fit(Xtrain, ytrain)
        f = open(spamhmm_path, 'wb')
        pickle.dump(spamhmm, f)
else:
    f = open(spamhmm_path, 'rb')
    spamhmm = pickle.load(f)

print('SpaMHMM trained/loaded!')

_1hmm_path = args.models_path + '/1hmm_h36m_' + args.action + '.pkl'
if args.train and args.train_1hmm:
    print('Training 1HMM...')
    _1hmm = SpaMHMM(n_nodes=K,
                    mix_dim=1,
                    n_components=get_S_1HMM(K, M, S, D),
                    n_features=D,
                    graph=None,
                    n_iter=args.n_iter,
                    verbose=args.train_verbose,
                    name='1hmm')

    _1hmm.fit(Xtrain, ytrain)
    f = open(_1hmm_path, 'wb')
    pickle.dump(_1hmm, f)
else:
    f = open(_1hmm_path, 'rb')
    _1hmm = pickle.load(f)

print('1HMM trained/loaded!')


khmm_path = args.models_path + '/khmm_h36m_' + args.action + '.pkl'
if args.train and args.train_khmm:
    print('Training KHMM...')
    khmm = KHMM(n_nodes=K,
                n_components=get_S_KHMM(K, M, S, D),
                n_features=D,
                verbose=args.train_verbose,
                name='khmm')

    khmm.fit(Xtrain, ytrain)
    f = open(khmm_path, 'wb')
    pickle.dump(khmm, f)
else:
    f = open(khmm_path, 'rb')
    khmm = pickle.load(f)

print('KHMM trained/loaded!')

print()
print('1HMM results')
print('n_components', _1hmm.n_components)
print('train loss', -_1hmm.score(Xtrain, ytrain))
print('test loss', -_1hmm.score(Xtest, ytest))
data_utils.predict(_1hmm, Xforecast_pref, Xforecast_suff, args.n_samples,
                   data_stats, pruned_joints, padded_dims, args.action)

print()
print('KHMM results')
print('n_components', khmm.n_components)
print('train loss', -khmm.score(Xtrain, ytrain))
print('test loss', -khmm.score(Xtest, ytest))
data_utils.predict(khmm, Xforecast_pref, Xforecast_suff, args.n_samples,
                   data_stats, pruned_joints, padded_dims, args.action)

print()
print('MHMM results')
print('mix_dim', mhmm.mix_dim)
print('n_components', mhmm.n_components)
mhmm_sparsity = np.sum(np.isclose(mhmm.mixCoef, 0., atol=1e-3))
print('absolute sparsity {0}, relative sparsity {1:.3f}'
      .format(mhmm_sparsity,
              mhmm_sparsity / (mhmm.n_nodes * mhmm.mix_dim)))
print('train loss', -mhmm.score(Xtrain, ytrain))
print('test loss', -mhmm.score(Xtest, ytest))
data_utils.predict(mhmm, Xforecast_pref, Xforecast_suff, args.n_samples,
                   data_stats, pruned_joints, padded_dims, args.action)

print()
print('SpaMHMM results')
print('mix_dim', spamhmm.mix_dim)
print('n_components', spamhmm.n_components)
print('reg', np.max(spamhmm.graph)/np.max(A))
spamhmm_sparsity = np.sum(np.isclose(spamhmm.mixCoef, 0., atol=1e-3))
print('absolute sparsity {0}, relative sparsity {1:.3f}'
      .format(spamhmm_sparsity,
              spamhmm_sparsity / (spamhmm.n_nodes * spamhmm.mix_dim)))
print('train loss', -spamhmm.score(Xtrain, ytrain))
print('test loss', -spamhmm.score(Xtest, ytest))
data_utils.predict(spamhmm, Xforecast_pref, Xforecast_suff, args.n_samples,
                   data_stats, pruned_joints, padded_dims, args.action)
