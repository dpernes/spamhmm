"""
Large portions of this file were borrowed or adapted from
https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
which is part of the source code for the paper:

Ashesh Jain, et al. "Structural-RNN: Deep learning on spatio-temporal graphs."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.
"""


import numpy as np
import copy

global rng
rng = np.random.RandomState(1234567890)

def runall(path_to_dataset, action, seq_len, seq_overlap, motion_prefix, motion_suffix):
    actions = [action]
    subactions=['1','2']
    trainSubjects = ['S1','S6','S7','S8','S9','S11']
    testSubject = ['S5']
    delta_shift = seq_len - seq_overlap

    #Load training and validation data
    [trainData,completeData]=loadTrainData(path_to_dataset, trainSubjects, actions, subactions)
    [testData,completeTestData]=loadTrainData(path_to_dataset, testSubject, actions, subactions)

    #Compute training data mean
    [data_mean,data_std,dimensions_to_ignore,new_idx]=normalizationStats(completeData)
    data_stats = {}
    data_stats['mean'] = data_mean
    data_stats['std'] = data_std
    data_stats['ignore_dimensions'] = dimensions_to_ignore

    #Create normalized 3D tensor for training and test
    [data3Dtensor,Y3Dtensor,data3Dtensor_t_1,minibatch_size] = sampleTrainSequences(trainData, data_mean, data_std, seq_len, delta_shift)
    [test3Dtensor,testY3Dtensor,test3Dtensor_t_1,minibatch_size_ignore] = sampleTrainSequences(testData, data_mean, data_std, seq_len, delta_shift)

    data3Dtensor = data3Dtensor.transpose(1,0,2)
    test3Dtensor = test3Dtensor.transpose(1,0,2)

    data3Dtensor, pruned_dims, padded_dims = pruneData(data3Dtensor, dimensions_to_ignore)
    test3Dtensor, _, _ = pruneData(test3Dtensor, dimensions_to_ignore)

    pruned_joints = [i//3 for i in pruned_dims if (i%3 == 0)]

    print('Training data stats (N,T,D) is ',data3Dtensor.shape)
    print('Test data stats (N,T,D) is ',test3Dtensor.shape)

    forecast_pref, _, forecast_suff, _ = generateForecastingExamples(testData,
                                                                     motion_prefix, motion_suffix,
                                                                     testSubject[0], actions, subactions,
                                                                     data_mean, data_std)

    return data3Dtensor, test3Dtensor, data_stats, forecast_pref, forecast_suff, pruned_joints, padded_dims

def normalizeTensor(inputTensor, data_mean, data_std):
    meanTensor = data_mean.reshape((1,1,inputTensor.shape[2]))
    meanTensor = np.repeat(meanTensor,inputTensor.shape[0],axis=0)
    meanTensor = np.repeat(meanTensor,inputTensor.shape[1],axis=1)
    stdTensor = data_std.reshape((1,1,inputTensor.shape[2]))
    stdTensor = np.repeat(stdTensor,inputTensor.shape[0],axis=0)
    stdTensor = np.repeat(stdTensor,inputTensor.shape[1],axis=1)
    normalizedTensor = np.divide((inputTensor - meanTensor),stdTensor)
    return normalizedTensor

def normalizationStats(completeData):
    data_mean = np.mean(completeData,axis=0)
    data_std =    np.std(completeData,axis=0)
    dimensions_to_ignore = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    new_idx = []
    count = 0
    for i in range(completeData.shape[1]):
        if i in dimensions_to_ignore:
            new_idx.append(-1)
        else:
            new_idx.append(count)
            count += 1

    '''Returns the mean of data, std, and dimensions with small std. Which we later ignore.    '''
    return data_mean,data_std,dimensions_to_ignore,np.array(new_idx)

def sampleTrainSequences(trainData, data_mean, data_std, T, delta_shift):
    training_data = []
    t_minus_one_data = []
    Y = []
    N = 0
    for k in trainData.keys():

        if len(k) == 3:
            continue

        data = trainData[k]
        start = 16
        end = T + 16
        while end + 1 < data.shape[0]:
            training_data.append(data[start:end,:])
            t_minus_one_data.append(data[start-1:end-1,:])
            Y.append(data[start+1:end+1,:])
            N += 1
            start += delta_shift
            end += delta_shift
    D = training_data[0].shape[1]
    data3Dtensor = np.zeros((T,N,D),dtype=np.float32)
    data3Dtensor_t_1 = np.zeros((T,N,D),dtype=np.float32)
    Y3Dtensor = np.zeros((T,N,D),dtype=np.float32)
    count = 0
    for x,y,t_1 in zip(training_data,Y,t_minus_one_data):
        data3Dtensor[:,count,:] = x
        data3Dtensor_t_1[:,count,:] = t_1
        Y3Dtensor[:,count,:] = y
        count += 1

    # Normalizing the training data features
    data3Dtensor_t_1 = normalizeTensor(data3Dtensor_t_1, data_mean, data_std) #np.divide((data3Dtensor - meanTensor),stdTensor)
    data3Dtensor = normalizeTensor(data3Dtensor, data_mean, data_std) #np.divide((data3Dtensor - meanTensor),stdTensor)
    Y3Dtensor = normalizeTensor(Y3Dtensor, data_mean, data_std) #np.divide((Y3Dtensor - meanTensor),stdTensor)
    return data3Dtensor,Y3Dtensor,data3Dtensor_t_1,N

def sampleConnectedTrainSequences(trainData, data_mean, data_std, T, delta_shift):
    training_data = []
    Y = []
    N = 0
    start= 0
    end = T
    minibatch_size = 0

    training_keys = trainData.keys()
    for k in training_keys:
        if len(k) < 4:
            continue
        if not k[3] == 'even':
            continue
        minibatch_size += 1


    while(True):
        isEnd = True
        for k in training_keys:

            if len(k) < 4:
                continue
            if not k[3] == 'even':
                continue

            data = trainData[k]
            fae = np.zeros((T,data.shape[1]),dtype=np.float32)
            labels = np.zeros((T,data.shape[1]),dtype=np.float32)

            if end + 1 < data.shape[0]:
                isEnd = False
                fea = data[start:end,:]
                labels = data[start+1:end+1,:]
            training_data.append(fea)
            Y.append(labels)
            N += 1
        if isEnd:
            break
        start += delta_shift
        end += delta_shift
    D = training_data[0].shape[1]
    data3Dtensor = np.zeros((T,N,D),dtype=np.float32)
    Y3Dtensor = np.zeros((T,N,D),dtype=np.float32)
    count = 0
    for x,y in zip(training_data,Y):
        data3Dtensor[:,count,:] = x
        Y3Dtensor[:,count,:] = y
        count += 1
    meanTensor = data_mean.reshape((1,1,data3Dtensor.shape[2]))
    meanTensor = np.repeat(meanTensor,data3Dtensor.shape[0],axis=0)
    meanTensor = np.repeat(meanTensor,data3Dtensor.shape[1],axis=1)
    stdTensor = data_std.reshape((1,1,data3Dtensor.shape[2]))
    stdTensor = np.repeat(stdTensor,data3Dtensor.shape[0],axis=0)
    stdTensor = np.repeat(stdTensor,data3Dtensor.shape[1],axis=1)

    # Normalizing the training data features
    data3Dtensor = normalizeTensor(data3Dtensor, data_mean, data_std) #np.divide((data3Dtensor - meanTensor),stdTensor)
    Y3Dtensor = normalizeTensor(Y3Dtensor, data_mean, data_std) #np.divide((Y3Dtensor - meanTensor),stdTensor)
    return data3Dtensor,Y3Dtensor,minibatch_size


def loadTrainData(path_to_dataset, subjects, actions, subactions):
    trainData = {}
    completeData = []
    for subj in subjects:
        for action in actions:
            for subact in subactions:
                filename = '{0}/{1}/{2}_{3}.txt'.format(path_to_dataset,subj,action,subact)
                action_sequence = readCSVasFloat(filename)

                T = action_sequence.shape[0]
                odd_list = range(1,T,2)
                even_list = range(0,T,2)

                trainData[(subj,action,subact)] = action_sequence
                trainData[(subj,action,subact,'even')] = action_sequence[even_list,:]
                trainData[(subj,action,subact,'odd')] = action_sequence[odd_list,:]
                if len(completeData) == 0:
                    completeData = copy.deepcopy(trainData[(subj,action,subact)])
                else:
                    completeData = np.append(completeData,trainData[(subj,action,subact)],axis=0)
    return trainData,completeData

def pruneData(data, dims_to_ignore):
    N, T, D = data.shape
    dims_to_prune = []
    dims_to_pad = []
    pruned_data = data.copy()
    for i in range(0, D, 3):
        if set(range(i, i+3)).issubset(set(dims_to_ignore)):
            dims_to_prune.extend(range(i, i+3))
            continue

        if i in dims_to_ignore:
            pruned_data[:, :, i] = np.random.standard_normal(data[:, :, i].shape)
            dims_to_pad.append(i)

        if (i + 1) in dims_to_ignore:
            pruned_data[:, :, i + 1] = np.random.standard_normal(data[:, :, i + 1].shape)
            dims_to_pad.append(i + 1)

        if (i + 2) in dims_to_ignore:
            pruned_data[:, :, i + 2] = np.random.standard_normal(data[:, :, i + 2].shape)
            dims_to_pad.append(i + 2)

    dims_to_keep = list(set(range(D)) - set(dims_to_prune))
    pruned_data = pruned_data[:, :, dims_to_keep]

    return pruned_data, dims_to_prune, dims_to_pad

def generateForecastingExamples(trainData, prefix, suffix, subject, actions, subactions, data_mean, data_std):
    N = 4*len(actions)*len(subactions)
    D = trainData[(subject,actions[0],subactions[0])].shape[1]
    trX = np.zeros((prefix,N,D),dtype=np.float32)
    trX_t_1 = np.zeros((prefix,N,D),dtype=np.float32)
    trY = np.zeros((suffix,N,D),dtype=np.float32)
    count = 0
    forecastidx = {}
    for action in actions:
        for i in range(4):
            for subact in subactions:
                data_to_use = trainData[(subject,action,subact,'even')]

                T = data_to_use.shape[0]
                idx = rng.randint(16,T-prefix-suffix)
                trX[:,count,:] = data_to_use[idx:(idx+prefix),:]
                trX_t_1[:,count,:] = data_to_use[idx-1:(idx+prefix-1),:]
                trY[:,count,:] = data_to_use[(idx+prefix):(idx+prefix+suffix),:]
                forecastidx[count] = (action,subact,idx)
                count += 1
    toget = 8
    if toget > count:
        toget = count
    return normalizeTensor(trX[:,:toget,:], data_mean, data_std), normalizeTensor(trX_t_1[:,:toget,:], data_mean, data_std),normalizeTensor(trY[:,:toget,:], data_mean, data_std),forecastidx

def writeMatToCSV(mat,filename):
    f = open(filename,'w')
    N = mat.shape[0]
    D = mat.shape[1]

    for i in range(N):
        st = ''
        for j in range(D):
            st += str(mat[i,j]) + ','
        st = st[:-1]
        f.write(st+'\n')

    f.close()

def readCSVasFloat(filename):
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))
    return np.array(returnArray)

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore):
    T = normalizedData.shape[0]
    D = data_mean.shape[0]
    origData = np.zeros((T,D),dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if not len(dimensions_to_use) == normalizedData.shape[1]:
        return []

    origData[:,dimensions_to_use] = normalizedData

    stdMat = data_std.reshape((1,D))
    stdMat = np.repeat(stdMat,T,axis=0)
    meanMat = data_mean.reshape((1,D))
    meanMat = np.repeat(meanMat,T,axis=0)
    origData = np.multiply(origData,stdMat) + meanMat

    return origData

def joint2node(joint_idx, pruned_joints):
    if joint_idx in pruned_joints:
        return None

    count_pruned = sum([joint < joint_idx for joint in pruned_joints])
    node_idx = joint_idx - count_pruned

    return node_idx

def node2joint(node_idx, pruned_joints):
    for joint_idx in range(33):
        if node_idx == joint2node(joint_idx, pruned_joints):
            return joint_idx

    return None

def saveForecastError(skel_err,err_per_dof,path,fname):
        f = open('{0}{1}'.format(path,fname),'w')
        for i in range(skel_err.shape[0]):
            f.write('T={0} {1}, {2}\n'.format(i,skel_err[i],err_per_dof[i]))
        f.close()