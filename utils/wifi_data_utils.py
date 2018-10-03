import os.path
import numpy as np
from pandas import read_csv
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA


def build_streams(file_name):
    '''
    Builds one stream per AP and per day. Does not discard any data.
    Returns:
        seqs_per_ap - a list of lists where seqs_per_ap[i][j] is an np.array
                      containing the sequence that occured in AP i on day j.
    '''
    if not os.path.isfile(file_name):
        return

    df = read_csv(file_name, names=['EnvironmentID', 'Year', 'Month', 'Day',
                                    'Hour', 'UserCount', 'SessionCount',
                                    'Duration', 'InputOctet', 'OutputOctet',
                                    'InputPacket', 'OutputPacket'], header=0)

    seqs_per_ap = []
    df_access_points = df[['EnvironmentID']].drop_duplicates()
    df_keys = df[['Year', 'Month', 'Day']].drop_duplicates()
    df_grp = df.groupby(['EnvironmentID', 'Year', 'Month', 'Day'])
    for ap in df_access_points.itertuples(index=False):
        ap_seq = []
        for key in df_keys.itertuples(index=False):
            if (ap + key) in df_grp.groups:
                seq = df_grp.get_group(ap + key).as_matrix(['UserCount',
                                                            'SessionCount',
                                                            'Duration',
                                                            'InputOctet',
                                                            'OutputOctet',
                                                            'InputPacket',
                                                            'OutputPacket'])
                ap_seq.append(seq)

        seqs_per_ap.append(ap_seq)

    return seqs_per_ap


def get_X_y_from_data(data):
    X = []
    y = []
    for ap, seq_list in enumerate(data):
        for seq in seq_list:
            y.append(ap)
            X.append(seq)

    return X, y


def get_data(data_dir, nets, csv_name, anomalies=None):
    X = []
    y = []
    anomaly_data = []
    for net in nets:
        net_path = data_dir + '/' + net
        filenames = os.listdir(net_path)
        subdirs = []
        for filename in filenames:  # loop through all the files and folders
                if os.path.isdir(os.path.join(os.path.abspath(net_path),
                                              filename)):
                        subdirs.append(filename)
        subdirs.sort()
        for subdir in subdirs:
            file_name = net_path + '/' + subdir + '/' + csv_name
            data = build_streams(file_name)
            Xsubdir, ysubdir = get_X_y_from_data(data)
            X.extend(Xsubdir)
            y.extend(ysubdir)
            if anomalies is not None:
                anomaly_subdir = [True
                                  if (net, yi) in anomalies
                                  else False
                                  for yi in ysubdir]
                anomaly_data.extend(anomaly_subdir)

    return X, y, anomaly_data


def preprocess(X, train=True, n_components=None, scaler=None, pca=None):
    if train:
        # subtract the mean and divide by the standard deviation
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        # apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X = pca.transform(X)

        return X, scaler, pca

    else:
        # subtract the mean and divide by
        # the standard deviation of the training set
        X = scaler.transform(X)

        # apply PCA for dimensionality reduction
        X = pca.transform(X)

        return X


def build_adj_matrix(dist_matrix_path, eps=5e-1):
    dist_matrix = np.genfromtxt(dist_matrix_path, delimiter=',')
    K = dist_matrix.shape[0]
    G = np.array([[1./dist_matrix[i, j] if i != j
                   else 0
                   for j in range(K)] for i in range(K)])

    if (G != G.T).any():
        G += G.T

    G = G / (1.0*np.max(G))
    G[G < eps] = 0.

    return G


def positive_rates(scores_neg, scores_pos, thr):
    true_negatives = np.sum(scores_neg >= thr)
    false_positives = np.sum(scores_neg < thr)
    true_positives = np.sum(scores_pos < thr)
    false_negatives = np.sum(scores_pos >= thr)

    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)

    return fpr, tpr


def get_auc(mdl, Xneg, yneg, Xpos, ypos):
    scores_neg = mdl.scores_per_seq(Xneg, yneg)
    scores_pos = mdl.scores_per_seq(Xpos, ypos)

    min_neg = np.min(scores_neg)
    max_neg = np.max(scores_neg)
    min_pos = np.min(scores_pos)
    max_pos = np.max(scores_pos)

    if min_neg < min_pos:
        score_min = min_neg
    else:
        score_min = min_pos

    if max_pos > max_neg:
        score_max = max_pos
    else:
        score_max = max_neg

    fpr = []
    tpr = []
    for thr in np.linspace(score_min, score_max, int(1e4)):
        fpr_i, tpr_i = positive_rates(scores_neg, scores_pos, thr)
        fpr.append(fpr_i)
        tpr.append(tpr_i)

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr
