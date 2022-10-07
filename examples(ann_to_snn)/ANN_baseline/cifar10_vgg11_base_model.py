import torch.backends.cudnn as cudnn
import time
import os

import torch.optim.lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CUDA configuration

from models.cifar10.vgg11 import *
from data.data_loader_cifar10 import get_train_valid_loader, get_test_loader
from utils.classification import training, testing
from utils.lib import dump_json, set_seed

set_seed(1111)

# Load datasets
home_dir = '/home/yangqu/MyProjects/Local_Tandem_Learning/LTL_CIFAR' # relative path
data_dir = '/home/yangqu/data' # Data dir
ckp_dir = os.path.join(home_dir, 'exp/cifar10/')

batch_size = 64
num_workers = 4

(train_loader, val_loader) = get_train_valid_loader(data_dir, batch_size=batch_size, num_workers=num_workers)
test_loader = get_test_loader(data_dir, batch_size=batch_size, num_workers=num_workers)

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	# Parameters
	num_epochs = 200
	global best_acc 
	best_acc = 0
	test_acc_history = []
	train_acc_history = []

	# Models and training configuration
	model = CNN_ReluX()
	model = model.to(device)
	cudnn.benchmark = True	

	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

	for epoch in range(num_epochs):
		since = time.time()

		# Training Stage
		model, acc_train, loss_train = training(model, train_loader, optimizer, criterion, device)


		# Testing Stage
		acc_test, loss_test = testing(model, test_loader, criterion, device)

		# log results
		test_acc_history.append(acc_test)
		train_acc_history.append(acc_train)

		scheduler.step()

		# Training Record
		time_elapsed = time.time() - since
		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60, time_elapsed % 60))
		print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
		print('Test Accuracy: {:4f}'.format(acc_test))

		# Save Model
		if acc_test > best_acc:
			print("Saving the model.")\

			if not os.path.isdir(ckp_dir+'checkpoint'):
				os.makedirs(ckp_dir+'checkpoint')

			state = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss_train,
					'acc': acc_test,
			}
			torch.save(state, ckp_dir+'checkpoint/vgg11_relu2_baseline_sgd.pt')
			best_acc = acc_test

	training_record = {
		'test_acc_history': test_acc_history,
		'train_acc_history': train_acc_history,
		'best_acc': best_acc,
	}
	dump_json(training_record, ckp_dir, 'vgg11_baseline_train_record_relu2_sgd')