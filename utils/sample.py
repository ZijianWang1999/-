import numpy as np

def noniid_train(dataset, num_users):
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    ratio = np.array([[0.0 for j in range(10)] for i in range(num_users)])
    num_data = [200, 300, 400, 600, 800]
    num_class = [i for i in range(10)]

    labels = np.array(dataset.targets)

    # {label -> [idx]}
    bucket = [[] for i in range(10)]
    for i in range(len(labels)):
        bucket[labels[i]].append(i)

    for i in range(num_users):
        total = int(np.random.choice(num_data, 1))
        for _ in range(10):
            id = int(np.random.choice(num_class, 1))
            ratio[i][id] += 0.1
        for j in range(10):
            num = int(total * ratio[i][j])
            rand_idxs = np.random.choice(bucket[j], num, replace=False)
            dict_users_train[i] = np.concatenate((dict_users_train[i], rand_idxs), axis=0)

    # print(ratio[num_users - 1])
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
