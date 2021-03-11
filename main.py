import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from models.Nets import CNN
from models.Server import FedAvg
from utils.sample import noniid_train, noniid_test
from utils.options import args_parser

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # split dataset {user_id: [list of data index]}
    dict_users_train = noniid_train(train_set, args.num_users)
    dict_users_test = noniid_test(test_set, args.num_users, dict_users_train)

    # load global model
    net_glob = CNN(args=args).to(args.device)
    net_glob.train()

    # parameters
    w_glob = net_glob.state_dict()
    w_locals = []

    loss_train = []

    # meta-learning for global initial parameters
    for epoch in range(args.meta_epochs):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            client = Client(args=args, dataset=train_set, idxs=dict_users_train[idx])
            w, loss = client.reptile(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # save global model
    torch.save(net_glob.state_dict(), './save/model.pt')
    print('Global model saved...')

    # test each of clients
    test_acc = [0 for i in range(args.num_users)]
    test_loss = [0 for i in range(args.num_users)]
    for idx in range(args.num_users):
        # every time start with the same global parameters
        net_glob.load_state_dict(w_glob)
        client = Client(args=args, dataset=train_set, idxs=dict_users_train[idx])
        w_client = client.fine_tune(net=copy.deepcopy(net_glob).to(args.device))

        client = Client(args=args, dataset=test_set, idxs=dict_users_test[idx])
        net_glob.load_state_dict(w_client)
        test_acc[idx], test_loss[idx] = client.test(net=copy.deepcopy(net_glob))

        torch.save(net_glob.state_dict(), './save/model{}.pt'.format(idx))
        print('Global model saved...')
    