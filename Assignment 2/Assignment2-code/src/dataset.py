import torch
from torchvision.io import read_image
from torchvision import transforms

import cv2
import imageio
from PIL import Image

import numpy as np
import os


class AnimalA2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_flag=True, img_sz=224):
        # shape = 50x85
        self.predicate_matrix = np.array(np.genfromtxt('../predicate-matrix-binary.txt', dtype=int))
        self.class_dict = {}
        self.reverse_class_dict = {}
        self.train_classes = []
        self.test_classes = []
        self.train_images = []
        self.test_images = []
        self.train_labels = []
        self.test_labels = []

        self.data_flag = data_flag
        self.img_sz = img_sz

        class_file = open('../classes.txt', 'r')
        for line in class_file.readlines():
            line = line.strip()
            class_number, class_name = line.split('\t')
            self.class_dict[int(class_number) - 1] = class_name
            self.reverse_class_dict[class_name] = int(class_number) - 1
        class_file.close()

        train_class_file = open('../trainclasses.txt', 'r')
        for line in train_class_file.readlines():
            self.train_classes.append(line.strip())
        train_class_file.close()

        test_class_file = open('../testclasses.txt', 'r')
        for line in test_class_file.readlines():
            self.test_classes.append(line.strip())
        test_class_file.close()

        for root, directories, files in os.walk('/common/users/kcm161/Princeton-Assignment2/data/JPEGImages/'):
            for filename in files:
                if filename.split('_')[0] not in self.test_classes:
                    self.train_images.append(os.path.join(root, filename))
                    self.train_labels.append(filename.split('_')[0])

        root_dir = '/common/users/kcm161/Princeton-Assignment2/data/JPEGImages/'
        test_images_file = open('../test_images.txt', 'r')
        for line in test_images_file.readlines():
            self.test_images.append(os.path.join(root_dir, line.strip().split(' ')[0]))
            self.test_labels.append(line.strip().split(' ')[1])
        test_images_file.close()


    def __getitem__(self, index):
        if self.data_flag:
            img = Image.open(self.train_images[index])
            class_name = self.train_labels[index]
            class_index = int(self.reverse_class_dict[class_name])
            predicate = self.predicate_matrix[class_index, :]
        else:
            img = Image.open(self.test_images[index])
            class_name = self.test_labels[index]
            class_index = int(self.reverse_class_dict[class_name])
            predicate = self.predicate_matrix[class_index, :]
            img_path = self.test_images[index]

        if img.getbands()[0] == 'L':
            img = img.convert('RGB')

        img = transforms.Resize((self.img_sz, self.img_sz))(img)
        img = transforms.ToTensor()(img)

        if not self.data_flag:
            return img, class_name, class_index, predicate, img_path
        else:
            return img, class_name, class_index, predicate

    def __len__(self):
        if self.data_flag:
            return len(self.train_images)
        else:
            return len(self.test_images)

if __name__ == '__main__':
    a2 = AnimalA2Dataset(data_flag=True)
    # print(a2.test_classes)
    # print(a2.test_images)
    # print(a2.test_labels)
    # print(a2.predicate_matrix.shape, a2.predicate_matrix[15])
    # print(a2.class_dict)
    # print(a2.reverse_class_dict)
    # print(len(a2.test_images), len(a2.train_images))
    print(a2.__getitem__(1))