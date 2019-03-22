import numpy as np
import pickle
from spamhmm import SpaMHMM
from khmm import KHMM
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from utils import wifi_data_utils as data_utils
from utils.global_utils import np2lst, lst2np, reg_graph
from utils.global_utils import get_S_1HMM, get_S_KHMM
import argparse

parser = argparse.ArgumentParser(description='Script configurations.')
parser.add_argument('--data_dir', type=str, default='./wifi_data')
parser.add_argument('--models_path', type=str, default='./pretrained_models')
parser.add_argument('--distance_matrix_path', type=str,
                    default='./wifi_data/distance_matrix.csv')
parser.add_argument('--roc_path', type=str, default='.')
parser.add_argument('--pca_dim', type=int, default=3)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--train_mhmm', type=int, default=1)
parser.add_argument('--train_spamhmm', type=int, default=1)
parser.add_argument('--train_1hmm', type=int, default=1)
parser.add_argument('--train_khmm', type=int, default=1)
parser.add_argument('--train_verbose', type=int, default=0)
parser.add_argument('--mix_dim', type=int, default=15)
parser.add_argument('--n_components', type=int, default=10)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--n_iter_mstep', type=int, default=100)
parser.add_argument('--lr_mstep', type=float, default=5e-3)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)

nets_train = ['Net1']
nets_test = ['Net2', 'Net3', 'Net5', 'Net6']
anomalies_annotation = [('Net2', 3), ('Net3', 1), ('Net5', 1), ('Net6', 1)]

Xtrain, ytrain, _ = data_utils.get_data(args.data_dir, nets_train,
                                        'output.csv')
Xtest, ytest, anomalies = data_utils.get_data(args.data_dir, nets_test,
                                              'output.csv',
                                              anomalies_annotation)

Xanom = [Xtest[i] for i in range(len(anomalies)) if anomalies[i]]
yanom = [ytest[i] for i in range(len(anomalies)) if anomalies[i]]
Xnormal = [Xtest[i] for i in range(len(anomalies)) if not anomalies[i]]
ynormal = [ytest[i] for i in range(len(anomalies)) if not anomalies[i]]
Xtest = Xnormal
ytest = ynormal

print('{} normal sequences for training'.format(len(Xtrain)))
print('{} normal sequences for testing'.format(len(Xtest)))
print('{} anomalous sequences for testing'.format(len(Xanom)))
print()

K = len(set(ytrain))
M = 1
S = 1
D = args.pca_dim

Xtrain, ytrain, lengths_train = lst2np(Xtrain, ytrain)
Xtest, ytest, lengths_test = lst2np(Xtest, ytest)
Xanom, yanom, lengths_anom = lst2np(Xanom, yanom)

Xtrain, scaler, pca = data_utils.preprocess(Xtrain, train=True,
                                            n_components=args.pca_dim)
Xtest = data_utils.preprocess(Xtest, train=False, scaler=scaler, pca=pca)
Xanom = data_utils.preprocess(Xanom, train=False, scaler=scaler, pca=pca)

A = data_utils.build_adj_matrix(args.distance_matrix_path)

Xtrain, ytrain = np2lst(Xtrain, ytrain, lengths_train)
Xtest, ytest = np2lst(Xtest, ytest, lengths_test)
Xanom, yanom = np2lst(Xanom, yanom, lengths_anom)

mhmm_path = args.models_path + '/mhmm_wifi.pkl'
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
        pgrid = {'mix_dim': [5, 10, 15], 'n_components': [3, 5, 8, 10]}

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

spamhmm_path = args.models_path + '/spamhmm_wifi.pkl'
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
        pgrid = {'graph': [1e-4*A, 5e-4*A, 1e-3*A, 5e-3*A, 1e-2*A, 5e-2*A,
                           1e-1*A]}

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

_1hmm_path = args.models_path + '/1hmm_wifi.pkl'
if args.train and args.train_1hmm:
    if args.cv:
        print('Training 1HMM using CV...')
        _1hmm = SpaMHMM(n_nodes=K,
                        mix_dim=1,
                        n_components=1,
                        n_features=D,
                        graph=None,
                        n_iter=args.n_iter,
                        verbose=args.train_verbose,
                        name='1hmm')
        S_1hmm = get_S_1HMM(K, M, S, D)
        pgrid = {'n_components': [int(S_1hmm/2), int(2*S_1hmm/3), S_1hmm,
                                  int(3*S_1hmm/2), 2*S_1hmm]}

        _1hmmCV = GridSearchCV(_1hmm, pgrid,
                               cv=StratifiedKFold(3, shuffle=True), n_jobs=-1)
        _1hmmCV.fit(Xtrain, ytrain)
        print('Best parameters')
        print(_1hmmCV.best_params_)
        f = open(_1hmm_path, 'wb')
        pickle.dump(_1hmmCV.best_estimator_, f)
        _1hmm = _1hmmCV.best_estimator_
    else:
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

khmm_path = args.models_path + '/khmm_wifi.pkl'
if args.train and args.train_khmm:
    if args.cv:
        print('Training KHMM using CV...')
        khmm = KHMM(n_nodes=K,
                    n_components=1,
                    n_features=D,
                    verbose=args.train_verbose,
                    name='khmm')
        S_khmm = get_S_KHMM(K, M, S, D)
        pgrid = {'n_components': [int(S_khmm/2), int(2*S_khmm/3), S_khmm,
                                  int(3*S_khmm/2), 2*S_khmm]}

        khmmCV = GridSearchCV(khmm, pgrid, cv=StratifiedKFold(3, shuffle=True),
                              n_jobs=-1)
        khmmCV.fit(Xtrain, ytrain)
        print('Best parameters')
        print(khmmCV.best_params_)
        f = open(khmm_path, 'wb')
        pickle.dump(khmmCV.best_estimator_, f)
        khmm = khmmCV.best_estimator_
    else:
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
print('test score', _1hmm.score(Xtest, ytest))
print('anomalies score', _1hmm.score(Xanom, yanom))
_1hmm_auc, _1hmm_fpr, _1hmm_tpr = data_utils.get_auc(_1hmm, Xtest, ytest,
                                                     Xanom, yanom)
print('AUC', _1hmm_auc)
np.savetxt('1hmm_roc.csv', np.concatenate([_1hmm_fpr.reshape(-1, 1),
                                           _1hmm_tpr.reshape(-1, 1)],
                                          axis=1), delimiter=',')

print()
print('KHMM results')
print('n_components', khmm.n_components)
print('test score', khmm.score(Xtest, ytest))
print('anomalies score', khmm.score(Xanom, yanom))
khmm_auc, khmm_fpr, khmm_tpr = data_utils.get_auc(khmm, Xtest, ytest, Xanom,
                                                  yanom)
print('AUC', khmm_auc)
np.savetxt('khmm_roc.csv', np.concatenate([khmm_fpr.reshape(-1, 1),
                                           khmm_tpr.reshape(-1, 1)],
                                          axis=1), delimiter=',')

print()
print('MHMM results')
print('mix_dim', mhmm.mix_dim)
print('n_components', mhmm.n_components)
mhmm_sparsity = np.sum(np.isclose(mhmm.mixCoef, 0.))
print('sparsity', mhmm_sparsity, 'rel',
      mhmm_sparsity/(mhmm.n_nodes * mhmm.mix_dim))
print('reg_obj', reg_graph(A, mhmm.mixCoef))
print('test score', mhmm.score(Xtest, ytest))
print('anomalies score', mhmm.score(Xanom, yanom))
mhmm_auc, mhmm_fpr, mhmm_tpr = data_utils.get_auc(mhmm, Xtest, ytest, Xanom,
                                                  yanom)
print('AUC', mhmm_auc)
np.savetxt('mhmm_roc.csv', np.concatenate([mhmm_fpr.reshape(-1, 1),
                                           mhmm_tpr.reshape(-1, 1)],
                                          axis=1),  delimiter=',')

print()
print('SpaMHMM results')
print('mix_dim', spamhmm.mix_dim)
print('n_components', spamhmm.n_components)
print('reg', spamhmm.graph[1, 0]/A[1, 0])
spamhmm_sparsity = np.sum(np.isclose(spamhmm.mixCoef, 0.))
print('sparsity', spamhmm_sparsity, 'rel',
      spamhmm_sparsity/(spamhmm.n_nodes * spamhmm.mix_dim))
print('reg_obj', reg_graph(A, spamhmm.mixCoef))
print('test score', spamhmm.score(Xtest, ytest))
print('anomalies score', spamhmm.score(Xanom, yanom))
spamhmm_auc, spamhmm_fpr, spamhmm_tpr = data_utils.get_auc(spamhmm, Xtest,
                                                           ytest, Xanom, yanom)
print('AUC', spamhmm_auc)
np.savetxt('spamhmm_roc.csv', np.concatenate([spamhmm_fpr.reshape(-1, 1),
                                              spamhmm_tpr.reshape(-1, 1)],
                                             axis=1), delimiter=',')
