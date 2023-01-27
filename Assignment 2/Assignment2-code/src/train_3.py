import argparse
import os
import sys
from tqdm import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from dataset import AnimalA2Dataset
from awa_classifier import resnet_awa, repvgg_awa

def train(model, train_loader, test_loader, optimizer, criterion, epoch_start, num_epochs, output_filename, distance_argument):    
    for epoch in range(epoch_start, epoch_start + num_epochs):
        with tqdm(train_loader, unit='batch') as loader:
            for itr, (img, _, indexes, predicates) in enumerate(loader):
                if img.shape[0] < 2:
                    break

                model.train()

                img = img.to(device)
                predicates = predicates.float().to(device)
                
                outputs = model(img)
                loss = criterion(outputs, predicates)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('batch_loss/train_loss', loss, itr * epoch)
                loader.set_postfix(loss=str(loss.item()))
                
            print('Finished training epoch {}'.format(epoch))

            accuracy = test(model, test_loader, distance_argument)
            print('Accuracy:', accuracy)
            print('Finished testing epoch {}'.format(epoch))

            writer.add_scalar('epoch_loss/training_loss', loss, epoch)
            writer.add_scalar('accuracy/test_accuracy', accuracy, epoch)
            
            torch.save({
                'state_dict':model.module.state_dict(),
				'train_loss': loss,
				'epoch': epoch}, 
				os.path.join('../checkpoints', f"model-waste-epoch-{str(epoch)}.pt"))


def calculate_cosine_distance(predicted_predicate, class_predicate):
	# return np.sum(curr_labels * class_labels) / np.sqrt(np.sum(curr_labels)) / np.sqrt(np.sum(class_labels))
    cosine_similarity = spatial.distance.cosine(predicted_predicate, class_predicate)
    return 1 - cosine_similarity


def calculate_euclidean_distance(predicted_predicate, class_predicate):
	# return np.sqrt(np.sum((curr_labels - class_labels)**2))
    return spatial.distance.euclidean(predicted_predicate, class_predicate)


def find_nearest_class(predicted_predicates, distance_argument):
	predictions = []
	for i in range(predicted_predicates.shape[0]):
		predicted_predicate = predicted_predicates[i,:].cpu().detach().numpy()
		best_index = float('-inf')
		best_dist = float('inf')

		for j in range(predicate_binary_mat.shape[0]):
			class_predicate = predicate_binary_mat[j,:]

			if distance_argument == 'euclidean':
				dist = calculate_euclidean_distance(predicted_predicate, class_predicate)
			elif distance_argument == 'cosine':
				dist = calculate_cosine_distance(predicted_predicate, class_predicate)

			if dist < best_dist and classes[j] not in train_classes:
				best_index = j
				best_dist = dist
		predictions.append(classes[int(best_index)])
		
	return predictions


def test(model, test_loader, distance_argument):
	model.eval()
	ground_truth = []
	preds = []
	with torch.no_grad():
		with tqdm(test_loader, unit='batch') as loader:	
			for i, (img, _, indexes, predicates, _) in enumerate(loader):
				img = img.to(device)
				predicates = predicates.float().to(device)
				outputs = model(img)

				gt = []
				for index in indexes:
					gt.append(classes[index.item()])
				ground_truth.extend(gt)

				pred = find_nearest_class(outputs, distance_argument)
				preds.extend(pred)

	preds = np.array(preds)
	ground_truth = np.array(ground_truth)
	accuracy = np.mean(pred_classes == ground_truth)

	return accuracy


def inference(model, test_loader, output_filename, distance_argument):
	model.eval()
	preds = []
	ground_truth = []
	img_paths = []

	attribute_wise_accuracy = [0] * 85  #threshold = 0.5

	with torch.no_grad():
		with tqdm(test_loader, unit='batch') as loader:
			for i, (img, _, indexes, predicates, img_path) in enumerate(loader):

				img = img.to(device)
				predicates = predicates.float().to(device)
				outputs = model(img)

				outputs_thresholded = np.where(outputs.cpu().detach().numpy() > 0.5, 1, 0)
				for row1, row2 in zip(predicates, outputs_thresholded):
					for idx in range(len(row1)):
						if row1[idx].item() == row2[idx]:
							attribute_wise_accuracy[idx] += 1

				gt=[]
				for index in indexes:
					gt.append(classes[index.item()])
				ground_truth.extend(gt)

				pred = find_nearest_class(outputs, distance_argument)
				preds.extend(pred)
				img_paths.extend(img_path)


	with open(output_filename, 'w') as f:
		for i in range(len(preds)):
			output_name = img_paths[i].replace('/common/users/kcm161/Princeton-Assignment2/data/JPEGImages/', '')
			f.write(output_name + ' ' + preds[i] + '\n')

	gt_number, pred_number = [], []
	for idx in range(len(ground_truth)):
		gt_number.append(reverse_class_dict[ground_truth[idx]] + 1)
		pred_number.append(reverse_class_dict[preds[idx]] + 1)

	fig, ax = plt.subplots()
	attribute_wise_accuracy = [x / len(preds) for x in attribute_wise_accuracy]
	plt.bar(attributes, attribute_wise_accuracy)
	plt.xticks(rotation='vertical')
	plt.show()

	fig, ax = plt.subplots(figsize=(20,20))
	cm = confusion_matrix(preds, ground_truth, labels=test_classes)
	disp = ConfusionMatrixDisplay(cm, display_labels=test_classes)
	disp.plot(xticks_rotation='vertical', ax=ax)
	plt.show()


def get_args() :
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_file", type=str, default='../output.txt')
    parser.add_argument("--distance", type=str, default='euclidean')
    parser.add_argument("--inference", type=bool, default=False)
    args = parser.parse_args()
    return args


args = get_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

save_dir = os.path.join('./logs/', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir)
writer = SummaryWriter(save_dir)

if args.model == 'resnet':
	model = resnet_awa()
elif args.model == 'repvgg':
	model = repvgg_awa()
	
if args.resume:
	checkpoint = torch.load(args.weights)
	model.load_state_dict(checkpoint['state_dict'])
	epoch = checkpoint['epoch'] + 1
else:
	epoch = 1
	
awa_object = AnimalA2Dataset(True)
train_classes = awa_object.train_classes
classes = awa_object.class_dict
predicate_binary_mat = awa_object.predicate_matrix
reverse_class_dict = awa_object.reverse_class_dict
test_classes = awa_object.test_classes
attributes = np.array(np.genfromtxt('../predicates.txt', dtype='str'))[:,-1]

train_dataset= AnimalA2Dataset(True)
test_dataset = AnimalA2Dataset(False)
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
criterion = nn.BCELoss()

model = nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if not args.inference:
	train(model, train_loader, test_loader, optimizer, criterion, epoch, args.num_epochs, args.output_file, args.distance)
if args.inference:
	inference(model, test_loader, args.output_file, args.distance)