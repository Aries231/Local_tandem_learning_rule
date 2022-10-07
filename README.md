# Local Tandem Learning (LTL)

The official code release for the Local Tandem Learning (LTL) framework.

# Usage
This repo contains the code for image classification on the Cifar10 dataset with the VGG11 and VGG16 architectures.

**Step 1. Pre-train baseline ANN models**

run ./examples(ann_to_snn)/ANN_baseline/cifar10_vgg11_base_model.py or cifar10_vgg16_base_model.py

**Step 2. Transfer to SNN model**

***a. Using offline version:***

run ./examples(ann_to_snn)/offline_LTL/cifar10_main_svgg11_offline.py or cifar10_main_svgg16_offline.py

***b. Using online version:***

run ./examples(ann_to_snn)/online_LTL/cifar10_main_svgg11_online.py or cifar10_main_svgg16_online.py

**Note:** 
Remember to change the 'home_dir' and 'data_dir' to the file path of your local machine
