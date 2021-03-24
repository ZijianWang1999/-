import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from models.Nets import CNNMnist, CNNCifar
from utils.sample import noniid_train2, noniid_test
from utils.options import args_parser
from models.Client import Client

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
        test_set = datasets.MNIST(root='./data/mnist', train=False, download=False, transform=transform)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10('./data/cifar', train=True, download=False, transform=transform)
        test_set = datasets.CIFAR10('./data/cifar', train=False, download=False, transform=transform)
    else:
        exit('Error: unrecognized dataset...')

    dict_users_train, ratio = noniid_train2(train_set, args.num_users)
    dict_users_test = noniid_test(test_set, args.num_users, ratio)

    print('Data finished...')

    # load global model
    net_glob = CNNCifar(args=args).to(args.device) if args.dataset == 'cifar' else CNNMnist(args=args).to(args.device)
    net_glob.train()

    # parameters
    w_glob = net_glob.state_dict()

    # test each of clients
    test_acc = [0 for i in range(args.num_users)]
    test_loss = [0 for i in range(args.num_users)]
    for idx in range(args.num_users):
        # every time start with the same global parameters
        net_glob.load_state_dict(w_glob)
        client = Client(args=args, dataset=train_set, idxs=dict_users_train[idx], bs=args.train_bs)
        w_client = client.local_train(net=copy.deepcopy(net_glob).to(args.device))

        client = Client(args=args, dataset=test_set, idxs=dict_users_test[idx], bs=args.test_bs)
        net_glob.load_state_dict(w_client)
        test_acc[idx], test_loss[idx] = client.test(net=copy.deepcopy(net_glob))

    print('\nAvg test acc = ({:.3f}%), Avg test loss = {:.3f}\n'.format(np.sum(np.array(test_acc)) / args.num_users
                                                                     , np.sum(np.array(test_loss)) / args.num_users))
    print('\nMin acc = {:.3f}, max acc = {:.3f}\n'.format(np.min(np.array(test_acc)), np.max(np.array(test_acc))))
    plt.figure()
    plt.plot(range(100), test_acc, 'o')
    plt.ylabel('Test accuracy')
    plt.xlabel('Client')
    plt.savefig('./save/local_train_test.png')