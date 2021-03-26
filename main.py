import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from models.Nets import CNNMnist, CNNCifar
from models.Server import FedAvg
from models.test import test_img
from utils.sample import noniid_train, noniid_test
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


    # split dataset {user_id: [list of data index]}
    dict_users_train, ratio = noniid_train(train_set, args.num_users)
    dict_users_test = noniid_test(test_set, args.num_users, ratio)

    print('Data finished...')

    # load global model
    net_glob = CNNCifar(args=args).to(args.device) if args.dataset == 'cifar' else CNNMnist(args=args).to(args.device)
    net_glob.train()

    # parameters
    w_glob = net_glob.state_dict()

    loss_train = []

    # meta-learning for global initial parameters
    for epoch in range(args.meta_epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            client = Client(args=args, dataset=train_set, idxs=dict_users_train[idx],bs=args.train_bs)
            w, loss = client.reptile(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if (epoch + 1) % 50 == 0:
            print('Round {:3d}, Average loss {:.3f}'.format(epoch + 1, loss_avg))
        loss_train.append(loss_avg)

        # print acc
        if (epoch + 1) % 100 == 0:
            acc_glob, loss_glob = test_img(net_glob, test_set, args)
            print('Epoch: {:3d} global accuracy: {:.3f}, global loss:{:.3f}'.format(epoch + 1, acc_glob, loss_glob))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('Train loss')
    plt.xlabel('Global epoch')
    plt.savefig('./save/fed_{}_C{}_{}.png'.format(args.meta_epochs, args.frac, args.dataset))

    # save global model
    torch.save(net_glob.state_dict(), './save/model.pt')
    print('Global model saved...')

    # test each of clients
    test_acc = [0 for i in range(args.num_users)]
    test_loss = [0 for i in range(args.num_users)]
    for idx in range(args.num_users):
        # every time start with the same global parameters
        net_glob.load_state_dict(w_glob)
        client = Client(args=args, dataset=train_set, idxs=dict_users_train[idx], bs=args.train_bs)
        w_client = client.fine_tune(net=copy.deepcopy(net_glob).to(args.device))

        client = Client(args=args, dataset=test_set, idxs=dict_users_test[idx], bs=args.test_bs)
        net_glob.load_state_dict(w_client)
        test_acc[idx], test_loss[idx] = client.test(net=copy.deepcopy(net_glob).to(args.device))

    print('\nAvg test acc = ({:.3f}%), Avg test loss = {:.3f}\n'.format(np.sum(np.array(test_acc)) / args.num_users
                                                                     , np.sum(np.array(test_loss)) / args.num_users))
    print('\nMin acc = {:.3f}, max acc = {:.3f}\n'.format(np.min(np.array(test_acc)), np.max(np.array(test_acc))))
    plt.figure()
    plt.plot(range(100), test_acc, 'o')
    plt.ylabel('Test accuracy')
    plt.xlabel('Client')
    plt.savefig('./save/fed_{}_C{}_{}_test.png'.format(args.meta_epochs, args.frac, args.dataset))