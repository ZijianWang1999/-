import numpy as np
from torchvision import datasets, transforms

def noniid_train(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # ratio [0...9] 
    ratio = [[0 for j in range(10)] for i in range(num_users)]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # get ratio
        for j in range(len(dict_users_train[i])):
            ratio[i][labels[dict_users_train[i][j]]] += 1
        ratio[i] /= sum(ratio[i])
    
    return dict_users_train, ratio

def noniid_test(dataset, num_users, ratio):
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    labels = dataset[:][1]
    bucket = [[] for i in range(10)]

    for i in range(labels.size(0)):
        bucket[labels[i][0]].append(i)

    for i in range(num_users):
        total = 100
        for j in range(9):
            num = int(ratio[i][j]*100)
            rand_idxs = np.random.choice(bucket[i], num, replace=False)
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)
            total -= num
        rand_idxs = np.random.choice(bucket[i], total, replace=False)
        dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)

    return dict_users_test