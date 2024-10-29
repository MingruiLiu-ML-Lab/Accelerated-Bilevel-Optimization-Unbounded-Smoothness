import torch
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
from methods import stocbio, saba, ma_soba, ttsa, bo_rep, accbo, sustain, vrbo
from data_loader import SNLIDataset, Sent140Dataset,  collate_pad, collate_pad_double
from torch.utils.data import DataLoader
import random
import numpy as np
torch.backends.cudnn.enabled = False
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='snli', type=str,
                        help="dataset: [news_data, snli, sentment140]", )

    parser.add_argument("--data_path", default='../data/dataset.json', type=str,
                        help="Path to dataset file")

    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch_size", )

    parser.add_argument("--save_direct", default='./logs', type=str,
                        help="Path to save file")

    parser.add_argument("--word2vec", default="data/wordvec.pkl", type=str,
                        help="Path to word2vec file")

    parser.add_argument("--methods" , default='stocbio', type=str,
                        help="choise method [stocbio, ttsa, saba, ma-soba, sustain, vrbo, bo-rep, accbo]")

    parser.add_argument("--epoch", default=20, type=int,
                        help="Number of outer iteration")
    
    parser.add_argument("--inner_batch_size", default=128, type=int,
                        help="Training batch size in inner iteration")

    parser.add_argument("--neumann_lr", default=1e-2, type=float,
                        help="update for hessian")

    parser.add_argument("--hessian_q", default=3, type=int,
                        help="Q steps for hessian-inverse-vector product")

    parser.add_argument("--outer_update_lr", default= 1e-1, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=1e-1, type=float,
                        help="Inner update learning rate")
    
    parser.add_argument("--inner_update_step", default=1, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--gamma", default=1e-3, type=float,
                        help="nesterov momentum")

    parser.add_argument("--seed", default=2, type=int,
                        help="random seed")

    parser.add_argument("--beta", default=0.90, type=float,
                        help="momentum parameters")

    parser.add_argument("--nu", default=1e-2, type=float,
                        help="learning rate of z")

    parser.add_argument("--y_warm_start", default=3, type=int,
                        help="update steps for y")

    parser.add_argument("--spider_loops", default=3, type=int,
                        help="the spider loops for vrbo")

    # RNN hyperparameter settings
    parser.add_argument("--word_embed_dim", default=300, type=int,
                        help="word embedding dimensions")

    parser.add_argument("--encoder_dim", default=4096, type=int,
                        help="encodding dimensions")

    parser.add_argument("--n_enc_layers", default=2, type=int,
                        help="encoding layers")

    parser.add_argument("--fc_dim", default=1024, type=int,
                        help="dimension of fully-connected layer")

    parser.add_argument("--n_classes", default=2, type=int,
                        help="classes of targets")

    parser.add_argument("--linear_fc", default=False, type=bool,
                        help="classes of targets")

    parser.add_argument("--pool_type", default="max", type=str,
                        help="type of pooling")

    parser.add_argument("--noise_rate", default=0.1, type=float,
                        help="rate for label noise")

    args = parser.parse_args()
    random_seed(args.seed)

    if args.data == 'snli':
        if os.path.isfile(f'data/{args.data}_train_{args.noise_rate}.pkl') and os.path.isfile(f'data/{args.data}_train_{args.noise_rate}.pkl'):
            print('Loading data...')
            train = torch.load(f'data/{args.data}_train_{args.noise_rate}.pkl')
            val = torch.load(f'data/{args.data}_dev_{args.noise_rate}.pkl')
            test = torch.load(f'data/{args.data}_test_{args.noise_rate}.pkl')
        else:
            print('Preprocessing data...')
            train = SNLIDataset("../data", "train", noise_rate=args.noise_rate)
            val = SNLIDataset("../data", "dev", noise_rate=args.noise_rate)
            test = SNLIDataset("../data", "test")
            torch.save(train, f'data/{args.data}_train_{args.noise_rate}.pkl')
            torch.save(train, f'data/{args.data}_dev_{args.noise_rate}.pkl')
            torch.save(test, f'data/{args.data}_test_{args.noise_rate}.pkl')
        training_size = train.dataset_size
        args.n_labels = 3
        args.n_classes = 3

    else:
        print('Do not support this data')
    st = time.time()

    if args.methods == 'stocbio':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 2e-3
        args.inner_update_step = 3
        learner = stocbio.Learner(args, training_size)

    elif args.methods == 'ttsa':
        args.outer_update_lr = 1e-3
        args.inner_update_lr = 1e-2
        learner = ttsa.Learner(args, training_size)


    elif args.methods == "saba":
        args.outer_update_lr = 5e-2
        args.inner_update_lr =2e-2
        args.nu = 1e-2
        learner = saba.Learner(args, training_size)

    elif args.methods == 'ma-soba':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-2
        args.beta = 0.9
        args.nu = 1e-2
        learner = ma_soba.Learner(args, training_size)


    elif args.methods == 'bo-rep':
        args.outer_update_lr = 2e-2
        args.inner_update_lr = 1e-2
        args.beta = 0.9
        args.nu = 1e-2
        args.inner_update_step = 3
        args.update_interval = 2
        learner = bo_rep.Learner(args, training_size)

    elif args.methods == 'sustain':
        args.outer_update_lr = 5e-2
        args.inner_update_lr = 5e-2
        args.beta = 0.9
        learner = sustain.Learner(args, training_size)

    elif args.methods == 'vrbo':
        args.outer_update_lr = 1e-1
        args.inner_update_lr = 5e-2
        args.spider_loops = 2
        args.update_interval = 2
        args.inner_batch_size = 256
        learner = vrbo.Learner(args, training_size)


    elif args.methods == 'accbo':
        args.outer_update_lr = 1e-1
        args.inner_update_lr = 1e-1
        args.beta = 0.9
        args.gamma = 0.1
        args.tau = 0.5
        args.update_interval = 2
        args.inner_update_step = 3
        learner = accbo.Learner(args, training_size)

    else:
        print('No such method, please change the method name!')

    global_step = 0
    acc_all_test = []
    acc_loss_test = []
    acc_all_train = []
    acc_loss_train = []
    for epoch in range(args.epoch):
        print(f"[epoch/epochs]:{epoch}/{args.epoch}")
        train_loader = DataLoader(train, shuffle=True, batch_size=args.inner_batch_size, collate_fn=collate_pad_double)
        val_loader = DataLoader(val, shuffle=True, batch_size=args.batch_size, collate_fn=collate_pad_double)
        test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_pad_double)
        acc, loss = learner(train_loader, val_loader, training=True, epoch=epoch)
        acc_all_train.append(acc)
        acc_loss_train.append(loss)
        print('training Loss:', acc_loss_train)
        print( 'training Acc:', acc_all_train)

        print("---------- Testing Mode -------------")

        acc, loss = learner.test(test_loader)
        acc_all_test.append(acc)
        acc_loss_test.append(loss)

        print(f'{args.methods} Test loss: {acc_loss_test}')
        print(f'{args.methods} Test Acc: {acc_all_test}')
        global_step += 1

    date = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    file_name = f'{args.methods}_outlr{args.outer_update_lr}_inlr{args.inner_update_lr}_seed{args.seed}_{date}'
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    save_path = 'logs/'

    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name)+'.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Train ACC': str(acc_all_train),
               'AVG Test ACC': str(acc_all_test), 'AVG Train LOSS': str(acc_loss_train), 'AVG Test LOSS': str(acc_loss_test),'time': total_time}))
    files.close()
    torch.save((acc_all_train, acc_all_test, acc_loss_train, acc_loss_test), os.path.join(save_path, file_name))
    print(f'time:{total_time} h')
    print(args)
if __name__ == "__main__":
    main()
