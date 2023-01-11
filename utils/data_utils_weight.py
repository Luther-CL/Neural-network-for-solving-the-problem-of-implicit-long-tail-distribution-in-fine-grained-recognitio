import logging

import torch
import os
import random
import json
from utils.my_dataset import MyDataSet
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler, \
    WeightedRandomSampler

logger = logging.getLogger(__name__)


def read_split_data_train(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        # val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images were found in the train dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    return train_images_path, train_images_label


def read_spilt_data_test(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        # val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the test dataset.".format(sum(every_class_num)))
    print("{} images for validation.".format(len(val_images_path)))
    return val_images_path, val_images_label


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    if args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.dataset == 'cub200':
        train_images_path, train_images_label = read_split_data_train(os.path.join(args.dataset_dir, "train"))
        val_images_path, val_images_label = read_spilt_data_test(os.path.join(args.dataset_dir, "test"))
        trainset = MyDataSet(images_path=train_images_path,
                             images_class=train_images_label,
                             transform=transform_train,
                             flag="train")

        # 实例化验证数据集
        testset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transform_test,
                            flag="test")

    if args.local_rank == 0:
        torch.distributed.barrier()

    # # 起初采样加权（70）
    # weight_records = [23, 25, 22, 22, 25, 23, 25, 24, 25, 24, 25, 23, 24, 25, 24, 23, 24, 25, 19, 25, 25, 25, 24, 25,
    #                   22, 22, 25, 18, 23, 25, 24, 23, 25, 24, 21, 25, 25, 25, 25, 25, 24, 25, 25, 25, 25, 22, 25, 24,
    #                   25, 25, 25, 25, 25, 24, 25, 25, 24, 5, 25, 22, 24, 19, 24, 19, 1, 24, 24, 20, 19, 25]

    # weight_records = [3, 1, 4, 4, 1, 3, 1, 2, 1, 2, 1, 3, 2, 1, 2, 3, 2, 1, 7, 1, 1, 1, 2, 1, 4, 4, 1, 8, 3, 1, 2, 3, 1,
    #                   2, 5, 1, 1, 1,
    #                   1, 1, 2, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 21, 1, 4, 2, 7, 2, 7, 25, 2, 2, 6, 7, 1]
    # for data, label, label_fine in trainset:
    #     # label = label.numpy()
    #     weights = [weight_records[label]]
    # train_sampler = WeightedRandomSampler(weights,num_samples=args.train_batch_size, replacement=True)

    # for data, label, label_fine in trainset:
    #     # label = label.numpy()
    # weights = [weight_records[label] for data, label, label_fine in trainset]
    # train_sampler = WeightedRandomSampler(weights, num_samples=args.train_batch_size, replacement=True)

    # # 随机采样
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

    # # 越是容易错分，就越要多采样训练。即200个类，细类所属粗类的细类数越多，该类就需采样越多
    # weight_records = [2, 1, 2, 1, 1, 1, 2, 2, 11, 1, 11, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 11, 3, 3, 3, 1, 11, 1, 11, 11, 3,
    #                   3, 3, 1, 1, 1, 7, 1, 7, 7, 1, 1, 7, 1, 8, 1, 1, 1, 11, 2, 2, 1, 1, 3, 1, 1, 1, 1, 8, 8, 8, 8, 1,
    #                   8, 8, 8, 3, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 1, 1,
    #                   3, 3, 3, 1, 1, 1, 7, 7, 1, 11, 1, 11, 11, 1, 1, 2, 2, 4, 1, 1, 7, 1, 7, 7, 7, 1, 1, 4, 3, 7, 3, 4,
    #                   3, 7, 7, 4, 1, 1, 1, 3, 2, 2, 3, 2, 2, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1,
    #                   3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 7, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                   1, 4, 4, 4, 1, 4, 1]

    # # 超过1的部分*0.5，steps+4000,若有效，再调系数
    # weight_records = [1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5, 1.5, 6.0, 1.0, 6.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
    #                   1.0, 1.0, 6.0, 2.0, 2.0, 2.0, 1.0, 6.0, 1.0, 6.0, 6.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 4.0, 1.0,
    #                   4.0, 4.0, 1.0, 1.0, 4.0, 1.0, 4.5, 1.0, 1.0, 1.0, 6.0, 1.5, 1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0,
    #                   1.0, 4.5, 4.5, 4.5, 4.5, 1.0, 4.5, 4.5, 4.5, 2.0, 2.0, 2.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0,
    #                   1.0, 2.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0,
    #                   2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 6.0, 1.0, 6.0, 6.0, 1.0, 1.0, 1.5, 1.5, 2.5, 1.0,
    #                   1.0, 4.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0, 2.5, 2.0, 4.0, 2.0, 2.5, 2.0, 4.0, 4.0, 2.5, 1.0, 1.0,
    #                   1.0, 2.0, 1.5, 1.5, 2.0, 1.5, 1.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #                   1.5, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #                   4.0, 1.0, 1.0, 1.5, 1.0, 2.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #                   1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 1.0, 2.5, 1.0]

    # weight_records = [10, 11, 10, 11, 11, 11, 10, 10, 1, 11, 1, 11, 11, 9, 9, 11, 11, 11, 11, 11, 11, 1, 9, 9, 9, 11, 1,
    #                   11, 1, 1, 9, 9, 9, 11, 11, 11, 5, 11, 5, 5, 11, 11, 5, 11, 4, 11, 11, 11, 1, 10, 10, 11, 11, 9,
    #                   11, 11, 11, 11, 4, 4, 4, 4, 11, 4, 4, 4, 9, 9, 9, 11, 10, 10, 11, 11, 11, 11, 11, 9, 11, 10, 11,
    #                   10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 1, 11, 11, 11, 9, 9, 9, 11, 11, 11, 5, 5, 11, 1, 11, 1, 1,
    #                   11, 11, 10, 10, 8, 11, 11, 5, 11, 5, 5, 5, 11, 11, 8, 9, 5, 9, 8, 9, 5, 5, 8, 11, 11, 11, 9, 10,
    #                   10, 9, 10, 10, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 10, 11, 11, 11, 10, 11, 11, 11, 9, 11, 11,
    #                   11, 11, 9, 11, 11, 11, 11, 11, 5, 11, 11, 10, 11, 9, 11, 10, 11, 11, 11, 10, 10, 11, 11, 11, 11,
    #                   11, 11, 11, 11, 11, 11, 8, 8, 8, 11, 8, 11]
    #
    # weights = [weight_records[label_fine] for data, label_coarse, label_fine in trainset]
    # # weights = weight_records
    # train_sampler = WeightedRandomSampler(weights, num_samples=args.num_classes, replacement=True)

    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=8,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader
