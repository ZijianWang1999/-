import numpy as np
from torchvision import datasets, transforms

def noniid_train(dataset, num_users, name):
    if name == 'cifar':
        num_shards, num_imgs = 200, 250
    else:
        num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # ratio [0...9] 
    ratio = np.array([[0.0 for j in range(10)] for i in range(num_users)])

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # get ratio
        for j in range(len(dict_users_train[i])):
            ratio[i][labels[dict_users_train[i][j]]] += 1.0
        ratio[i] = ratio[i] / np.sum(ratio[i])
    
    return dict_users_train, ratio

def noniid_test(dataset, num_users, ratio):
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    labels = np.array(dataset.targets)

    # {label -> [idx]}
    bucket = [[] for i in range(10)]
    for i in range(len(labels)):
        bucket[labels[i]].append(i)

    for i in range(num_users):
        total = 500
        for j in range(9):
            num = int(ratio[i][j]*500)
            rand_idxs = np.random.choice(bucket[j], num, replace=False)
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)
            total -= num
        rand_idxs = np.random.choice(bucket[9], total, replace=False)
        dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)

    return dict_users_test