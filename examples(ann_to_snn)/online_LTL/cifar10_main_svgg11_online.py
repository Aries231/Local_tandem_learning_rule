import time
import os
import argparse
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CUDA configuration
current_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(current_dir)

import torch
import torch.nn as nn
from models.cifar10.vgg11 import CNN_ReluX, SpikingVGG11OL
from data.data_loader_cifar10 import get_train_valid_loader, get_test_loader
from utils.classification import testing_snn_st, training_snn_st
from utils.lib import dump_json,set_seed

set_seed(1111)

# Load datasets
home_dir = '/home/yangqu/MyProjects/Local_Tandem_Learning/LTL_CIFAR' # relative path
data_dir = '/home/yangqu/data' # Data dir
ann_ckp_dir = os.path.join(home_dir, 'exp/cifar10/')
snn_ckp_dir = os.path.join(home_dir, 'exp/cifar10/snn/offline/')

parser = argparse.ArgumentParser(description='PyTorch Cifar-10 Training')
parser.add_argument('--Tencode', default=16, type=int, metavar='N',
                    help='encoding time window size')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save the training record (default: none)')
parser.add_argument('--local_coefficient', default=1.0, type=float,
                     help='Coefficient of Local Loss')
parser.add_argument('--Twarm', default=12, type=int,
                     help='warm up time')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    # Parameters
    args = parser.parse_args()
    Tencode = args.Tencode
    Twarm = args.Twarm
    num_epochs = args.epochs
    lr = args.lr
    best_test_acc = 0
    batch_size = args.batch_size
    p = 99
    #train_record_path = args.save_path
    train_record_path = 'spiking_vgg11_t{0}_Tw{1}_tau10_L2_p{1}_online_relu2'.format(Tencode, Twarm, p)
    coeff_local = [args.local_coefficient] * 10 # Local loss coefficient
    test_acc_history = []
    train_acc_history = []

    # Prepare the data
    (train_loader, val_loader) = get_train_valid_loader(data_dir, batch_size=batch_size, num_workers=4)
    test_loader = get_test_loader(data_dir, batch_size=batch_size, num_workers=4)

    # Init ANN and load pre-trained model
    ann = CNN_ReluX()
    ann = ann.to(device)
    checkpoint = torch.load(ann_ckp_dir + 'checkpoint/vgg11_relu2_baseline_sgd.pt')
    ann.load_state_dict(checkpoint['model_state_dict'])
    print('Accuracy of pre-trained model {}'.format(checkpoint['acc']))

    # Init SNN model
    snn = SpikingVGG11OL(Tencode).to(device)

    # Training configuration
    criterion_out = torch.nn.CrossEntropyLoss()
    criterion_local = nn.MSELoss()  # Local loss function
    optimizer = torch.optim.Adam(snn.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30,35,40,45], gamma=0.2)

    for epoch in range(num_epochs):
        since = time.time()

        # Training Stage
        snn, acc_train, loss_train = training_snn_st(ann, snn, train_loader, optimizer, criterion_out, criterion_local,\
                                                     coeff_local, device, Twarm, p)

        # Testing Stage
        acc_test, loss_test = testing_snn_st(snn, test_loader, criterion_out, device, Twarm)

        scheduler.step()

        # log results
        test_acc_history.append(acc_test)
        train_acc_history.append(acc_train)

        # Report Training Progress
        time_elapsed = time.time() - since
        print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
        print('Test Accuracy: {:4f}'.format(acc_test))

        # Save Model
        if acc_test > best_test_acc:
            print("Saving the model.")

            if not os.path.isdir(snn_ckp_dir + 'checkpoint'):
                os.makedirs(snn_ckp_dir + 'checkpoint')

            state = {
                'epoch': epoch,
                'model_state_dict': snn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                'acc': acc_test,
            }
            #torch.save(state, snn_ckp_dir + 'checkpoint/spiking_vgg11_t{0}_Tw{1}_tau10_L2_p{2}_online.pt'.format(Tencode, Twarm, p))
            best_test_acc = acc_test

    print('Best Test Accuracy: {:4f}'.format(best_test_acc))

    training_record = {
        'test_acc_history': test_acc_history,
        'train_acc_history': train_acc_history,
        'best_acc': best_test_acc,
    }
    dump_json(training_record, snn_ckp_dir, train_record_path)