Local Tandem Learning (LTL) rule:

1. Pre-train ANN model: run ./examples(ann_to_snn)/ANN_baseline/cifar10_vgg11_base_model.py or cifar10_vgg16_base_model.py

2. Transfer to SNN model
Using offline version: run ./examples(ann_to_snn)/offline_LTL/cifar10_main_svgg11_offline.py or cifar10_main_svgg16_offline.py
Using online version: run ./examples(ann_to_snn)/online_LTL/cifar10_main_svgg11_online.py or cifar10_main_svgg16_online.py

Note: 
Remember to change the 'home_dir' and 'data_dir' to the file path of your local machine

