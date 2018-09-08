"""Functions that help with data processing for human3.6m"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy
import sys
sys.path.append('./utils')
import h36m_processdata as poseDataset

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot ):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
  """
  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization
  Args
    poses: The output from the TF model. A list with (seq_length) entries,
    each with a (batch_size, dim) output
  Returns
    poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
    batch is an n-by-d sequence of poses.
  """
  seq_len = len(poses)
  if seq_len == 0:
    return []

  batch_size, dim = poses[0].shape

  poses_out = np.concatenate(poses)
  poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  poses_out = np.transpose(poses_out, [1, 0, 2])

  poses_out_list = []
  for i in xrange(poses_out.shape[0]):
    poses_out_list.append(
      unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

  return poses_out_list


def readCSVasFloat(filename):
  """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
  returnArray = []
  lines = open(filename).readlines()
  for line in lines:
    line = line.strip().split(',')
    if len(line) > 0:
      returnArray.append(np.array([np.float32(x) for x in line]))

  returnArray = np.array(returnArray)
  return returnArray


def load_data(path_to_dataset, subjects, actions, one_hot):
  """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270
  Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
  Returns
    trainData: dictionary with k:v
      k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
  """
  nactions = len( actions )

  trainData = {}
  completeData = []
  for subj in subjects:
    for action_idx in np.arange(len(actions)):

      action = actions[ action_idx ]

      for subact in [1, 2]:  # subactions

        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

        filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
        action_sequence = readCSVasFloat(filename)

        n, d = action_sequence.shape
        even_list = range(0, n, 2)

        if one_hot:
          # Add a one-hot encoding at the end of the representation
          the_sequence = np.zeros( (len(even_list), d + nactions), dtype=float )
          the_sequence[ :, 0:d ] = action_sequence[even_list, :]
          the_sequence[ :, d+action_idx ] = 1
          trainData[(subj, action, subact, 'even')] = the_sequence
        else:
          trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]


        if len(completeData) == 0:
          completeData = copy.deepcopy(action_sequence)
        else:
          completeData = np.append(completeData, action_sequence, axis=0)

  return trainData, completeData


def normalize_data( data, data_mean, data_std, dim_to_use, actions, one_hot ):
  """
  Normalize input data by removing unused dimensions, subtracting the mean and
  dividing by the standard deviation
  Args
    data: nx99 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    data_out: the passed data matrix, but normalized
  """
  data_out = {}
  nactions = len(actions)

  if not one_hot:
    # No one-hot encoding... no need to do anything special
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ]

  else:
    # TODO hard-coding 99 dimensions for un-normalized human poses
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key][:, 0:99] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ]
      data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )

  return data_out


def normalization_stats(completeData):
  """"
  Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
  Args
    completeData: nx99 matrix with data to normalize
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use
  
  
def build_adj_matrix():
  A = np.zeros((33, 33))
  
  # actual joint connections
  A[1, 2]   = 1
  A[1, 7]   = 1
  A[1, 12]  = 1
  A[2, 3]   = 1
  A[3, 4]   = 1
  A[4, 5]   = 1
  A[5, 6]   = 1
  A[7, 8]   = 1
  A[8, 9]   = 1
  A[9, 10]  = 1
  A[10, 11] = 1
  A[12, 13] = 1
  A[13, 17] = 1
  A[13, 14] = 1
  A[14, 15] = 1
  A[15, 16] = 1
  A[13, 25] = 1
  A[17, 18] = 1
  A[18, 19] = 1
  A[19, 20] = 1
  A[20, 21] = 1
  A[21, 22] = 1
  A[20, 23] = 1
  A[23, 24] = 1
  A[25, 26] = 1
  A[26, 27] = 1
  A[27, 28] = 1
  A[28, 29] = 1
  A[29, 30] = 1
  A[28, 31] = 1
  A[31, 32] = 1
  
  # symmetry connections (one side to the other)
  A[2, 7]   = 1
  A[3, 8]   = 1
  A[4, 9]   = 1
  A[5, 10]  = 1
  A[6, 11]  = 1
  A[17, 25] = 1
  A[18, 26] = 1
  A[19, 27] = 1
  A[20, 28] = 1
  A[21, 29] = 1
  A[22, 30] = 1
  A[23, 31] = 1
  A[24, 32] = 1
  
  for j in range(33):
    for i in range(j + 1, 33):
      A[i, j] = A[j, i]
      
  return A    


def build_inputs(data):
  if data is None:
    return None, None, None
  N, T, D = data.shape
  
  X = np.zeros((N * (D//3), T, 3))
  y = np.zeros(N * (D//3)).astype(int)
  for i in range(0, D, 3):
    X[N*i//3 : N*i//3 + N, :, :] = data[:, :, i : i +3]
    y[N*i//3 : N*i//3 + N] = i//3
  
  X = X.reshape(-1, 3)
  lengths = [T for i in range(N * (D//3))]
  
  return X, y, lengths  


def prune_adj_matrix(adj_matrix, joints_to_prune):
  A = adj_matrix.copy()
  A = np.delete(A, np.array(joints_to_prune).astype(int), axis=0)
  A = np.delete(A, np.array(joints_to_prune).astype(int), axis=1)
  
  return A


def predict(mdl, Xforecast_pref, Xforecast_suff, nsamp, data_stats, pruned_joints, padded_dims, action='walking'):
  # PREDICT USING SAMPLING
  Nseqs = Xforecast_pref.shape[1]
  suff_len = Xforecast_suff.shape[0]
  
  for seq_idx in range(Nseqs):
    Xseq_pref = Xforecast_pref[:, seq_idx, :]
    Xseq_suff = Xforecast_suff[:, seq_idx, :]
    seq_gt = np.array([]).reshape(suff_len, 0)
    seq_pred = np.array([]).reshape(suff_len, 0)
    for joint_idx in range(33):
      if joint_idx in pruned_joints:
        continue
      
      node_idx = poseDataset.joint2node(joint_idx, pruned_joints)
      
      keep_dims = [0, 1, 2]
      if 3*joint_idx in padded_dims:
        Xseq_pref[:, 3*joint_idx] = np.random.standard_normal(Xseq_pref[:, 3*joint_idx].shape)
        keep_dims.remove(0)
      if 3*joint_idx + 1 in padded_dims:
        Xseq_pref[:, 3*joint_idx + 1] = np.random.standard_normal(Xseq_pref[:, 3*joint_idx + 1].shape)
        keep_dims.remove(1)
      if 3*joint_idx + 2 in padded_dims:
        Xseq_pref[:, 3*joint_idx + 2] = np.random.standard_normal(Xseq_pref[:, 3*joint_idx + 2].shape)
        keep_dims.remove(2)
      
      Xgt = Xseq_suff[:, 3*joint_idx : 3*joint_idx + 3]
      Xgt = Xgt[:, keep_dims]
      
      Xpred = np.zeros((suff_len, 3))
      # if type(mdl) == list:
      #   for i in range(nsamp):
      #     Xpred += mdl[node_idx].sample(0, suff_len, Xpref=Xseq_pref[:, 3*joint_idx : 3*joint_idx + 3])[0]
      #   Xpred = Xpred[:, keep_dims]/nsamp  
      # else:
      for i in range(nsamp):
        Xpred += mdl.sample(node_idx, suff_len, Xpref=Xseq_pref[:, 3*joint_idx : 3*joint_idx + 3])[0]
      Xpred = Xpred[:, keep_dims]/nsamp
      
      seq_gt = np.concatenate([seq_gt, Xgt], axis=1)
      seq_pred = np.concatenate([seq_pred, Xpred], axis=1)
      
    seq_gt = poseDataset.unNormalizeData(seq_gt, data_stats['mean'], data_stats['std'], data_stats['ignore_dimensions'])
    seq_pred = poseDataset.unNormalizeData(seq_pred, data_stats['mean'], data_stats['std'], data_stats['ignore_dimensions'])
    
    if seq_idx == 0:
      all_gts = np.zeros((suff_len, Nseqs, seq_gt.shape[1]))
      all_gts[:, 0, :] = seq_gt
    
      all_preds = np.zeros((suff_len, Nseqs, seq_pred.shape[1]))
      all_preds[:, 0, :] = seq_pred
    else:
      all_gts[:, seq_idx, :] = seq_gt
      all_preds[:, seq_idx, :] = seq_pred
    
    print('Seq {} done'.format(seq_idx))
    
  mean_errors = np.zeros( (Nseqs, suff_len) )
  for i in range(Nseqs):
    eulerchannels_pred = all_preds[:,i,:]
    eulerchannels_gt = all_gts[:,i,:]
    
    # Convert from exponential map to Euler angles
    for j in np.arange( eulerchannels_pred.shape[0] ):
      for k in np.arange(3,97,3):
        eulerchannels_pred[j,k:k+3] = rotmat2euler(
          expmap2rotmat( eulerchannels_pred[j,k:k+3] ))
        eulerchannels_gt[j,k:k+3] = rotmat2euler(
          expmap2rotmat( eulerchannels_gt[j,k:k+3] ))  
    
    # The global translation (first 3 entries) and global rotation
    # (next 3 entries) are also not considered in the error, so the_key
    # are set to zero.
    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
    gt_i=np.copy(eulerchannels_gt)
    gt_i[:,0:6] = 0
    
    # Now compute the l2 error. The following is numpy port of the error
    # function provided by Ashesh Jain (in matlab), available at
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
    idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]
    
    euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
    euc_error = np.sum(euc_error, 1)
    euc_error = np.sqrt( euc_error )
    mean_errors[i,:] = euc_error
    
  # This is simply the mean error over the N_SEQUENCE_TEST examples
  mean_mean_errors = np.mean( mean_errors, 0 )
  
  # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
  print('{0: <16} |'.format(action), end='')
  for ms in [1,3,7,9,13,24]:
    print(' {0:.3f} |'.format( mean_mean_errors[ms] ), end='')  
  print()  