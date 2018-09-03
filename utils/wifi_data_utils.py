import os.path
import numpy as np
from pandas import read_csv
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# '''
# Builds one stream per AP and per day. Discards APs where data is missing. 
# Returns: 
#   streams - an N x A x L x F np.array containing all sequences for every AP. 
#   ap_ids - the IDs of all APs that have not been discarded.
# '''
# def build_balanced_streams(data_path):
#   if not os.path.isfile(data_path):
#     return
    
#   df = read_csv(data_path, names=['APid','Year','Month','Day','Hour','UserCount',
#         'SessionCount','Duration','InputOctet','OutputOctet','InputPacket',
#         'OutputPacket','F0','F1','F2','Bool'])
  
#   # remove APs for which we don't have 4 observations in every hour
#   df_keys = df[['Year','Month','Day','Hour']].drop_duplicates()
#   df_access_points = df[['APid']].drop_duplicates()
#   df_grp = df.groupby(['APid','Year','Month','Day','Hour'])
#   ap_to_del = []
#   for ap in df_access_points.itertuples(index=False):
#     for key in df_keys.itertuples(index=False):
#       if ap not in ap_to_del:
#         if (ap + key) in df_grp.groups:
#           L = len(df_grp.get_group(ap + key))
#           if L != 4:
#             ap_to_del.append(ap)
#         else:
#           ap_to_del.append(ap)
  
#   for ap in ap_to_del:
#     df.drop(df[df['APid'] == ap].index, inplace=True)
  
#   A = len(df[['APid']].drop_duplicates()) # number of access points after dropping
#   N = len(df[['Year','Month','Day']].drop_duplicates()) # number of training examples (days) for each AP
#   data = df.as_matrix(columns=['APid','F0','F1','F2'])
#   L = data.shape[0]//(A*N) # sequence length
#   streams = np.zeros((N,A,L,3))
#   ap_ids = np.zeros(A, dtype=int)
#   for i in range(A):
#     streams[:,i,:,:] = data[i*N*L : (i+1)*N*L, 1::].reshape(N,L,3)
#     ap_ids[i] = int(data[i*N*L, 0])
  
#   return streams, ap_ids

''' 
Builds one stream per AP and per day. Does not discard any data.
Returns:
  seqs_per_ap - a list of lists where seqs_per_ap[i][j] is an np.array 
                containing the sequence that occured in AP i on day j.
'''
def build_streams(file_name):
  if not os.path.isfile(file_name):
    return
   
  df = read_csv(file_name, names=['EnvironmentID','Year','Month','Day',
        'Hour','UserCount','SessionCount','Duration','InputOctet',
        'OutputOctet','InputPacket','OutputPacket'], header=0)
        
  seqs_per_ap = []
  df_access_points = df[['EnvironmentID']].drop_duplicates()
  df_keys = df[['Year','Month','Day']].drop_duplicates()
  df_grp = df.groupby(['EnvironmentID','Year','Month','Day'])
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
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(net_path), filename)): # check whether the current object is a folder or not
            subdirs.append(filename)
    subdirs.sort()
    for subdir in subdirs:
      file_name = net_path +'/'+ subdir +'/'+ csv_name
      data = build_streams(file_name)
      Xsubdir, ysubdir = get_X_y_from_data(data)
      X.extend(Xsubdir)
      y.extend(ysubdir)
      if anomalies is not None:
        anomaly_subdir = [True if (net, yi) in anomalies else False for yi in ysubdir]
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
    # subtract the mean and divide by the standard deviation of the training set
    X = scaler.transform(X)
    
    # apply PCA for dimensionality reduction
    X = pca.transform(X)
      
    return X


def build_adj_matrix(dist_matrix_path):
  dist_matrix = np.genfromtxt(dist_matrix_path, delimiter=',')
  K = dist_matrix.shape[0]
  G = np.array([[1./dist_matrix[i,j] if i != j
                 else 0 
                 for j in range(K)] for i in range(K)])
  
  if (G != G.T).any():
    G += G.T
    
  G = G / (1.0*np.max(G))  
  
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
  for thr in np.linspace(score_min, score_max, int(1e6)):
    fpr_i, tpr_i = positive_rates(scores_neg, scores_pos, thr)
    fpr.append(fpr_i)
    tpr.append(tpr_i)  
  
  plt.figure()
  plt.plot(fpr, tpr)
  plt.savefig(mdl.name + '_roc.png')
  
  auc = metrics.auc(fpr, tpr)
  
  return auc, fpr, tpr
