import json
import os
from collections import defaultdict, Counter
import pickle

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
import numpy as np


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, images, questions, answers, class_ids):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.class_ids = class_ids

    def __getitem__(self, index):
        return self.images[self.questions[index][0]], self.questions[index][1], self.class_ids[self.answers[index]]

    def __len__(self):
        return len(self.questions)


class VQAModel(nn.Module):
    def __init__(self, bert, num_classes=1000):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Linear(728, 512)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(2816, 1000)

    def forward(self, img_emb, question, predict=False):
        question_emb = self.bert(**question).last_hidden_state.to(device)
        if predict:
            question_emb = question_emb[0, 0, :]
            emb = torch.cat((img_emb, question_emb), 0)
        else:
            question_emb = question_emb[:, 0, :]
            emb = torch.cat((img_emb, question_emb), 1)
        emb = self.fc3(emb)
        return emb


class VQAImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(2048,512), nn.Dropout(0.25), nn.ReLU(), nn.Linear(512, 512))

    def forward(self, img):
        return self.fc1(img)


class VQATextModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Sequential(nn.Linear(768,512), nn.Dropout(0.25), nn.ReLU(),
                                 nn.Linear(512, 512))

    def forward(self, text):
        feature =  self.bert(**text).last_hidden_state.to(device)
        return self.fc1(feature[:, 0, :])


class VQAFinalModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, emb):
        return self.fc1(emb)


def train_add(loader, img_model, txt_model, final_model, criterion, optimizer, num_epochs=25):
    question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for epoch in range(num_epochs):
        print('Starting epoch {}'.format(epoch))
        img_model.train()
        txt_model.train()
        final_model.train()
        running_loss = 0.
        last_loss = 0.

        for i, (img, question, label) in enumerate(loader):
            if torch.cuda.is_available():
                img, label = img.to(device), label.float().to(device)
            
            question = question_tokenizer(question, max_length=30, padding='max_length', return_tensors = 'pt')
            question = question.to(device)

            img_emb = img_model(img).to(device)
            question_emb = txt_model(question).to(device)
            emb = torch.add(img_emb, question_emb)
            prediction = final_model(emb).to(device)

            predicted_idx = torch.argmax(prediction, dim=1)
            label_idx = torch.argmax(label, dim=1)
            loss = criterion(prediction, label_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0:
                last_loss = running_loss / 1000
                print('batch {} loss {}'.format(i+1, last_loss))
                running_loss = 0.
            
        print('Finished epoch {}'.format(epoch))

    torch.save(img_model.state_dict(), 'img_model-add.pt')
    torch.save(txt_model.state_dict(), 'txt_model-add.pt')
    torch.save(final_model.state_dict(), 'final_model-add.pt')


def train_concat(loader, model, criterion, optimizer, num_epochs=25):
    question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for epoch in range(num_epochs):
        print('Starting epoch {}'.format(epoch))

        model.train()
        running_loss = 0.
        last_loss = 0.

        for i, (img, question, label) in enumerate(loader):
            if torch.cuda.is_available():
                img, label = img.to(device), label.float().to(device)
            
            question = question_tokenizer(question, max_length=30, padding='max_length', return_tensors = 'pt')
            question = question.to(device)

            prediction = model(img, question).to(device)

            predicted_idx = torch.argmax(prediction, dim=1)
            label_idx = torch.argmax(label, dim=1)
            loss = criterion(prediction, label_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0:
                last_loss = running_loss / 1000
                print('batch {} loss {}'.format(i+1, last_loss))
                running_loss = 0.
            
        print('Finished epoch {}'.format(epoch))

    torch.save(model.state_dict(), 'model.pt')


def val_concat(val_questions, val_answers, val_images, model, classes):
    val_qs = []
    val_ans = []
    val_img = []
    results = []
    counter = 0

    model.eval()

    question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for (q, a) in zip(val_questions['questions'], val_answers['annotations']):
        val_ans.append(a['multiple_choice_answer'].replace(' ', ''))
        val_qs.append((q['question_id'], q['question']))
        val_img.append(val_images[q['image_id']])

    with torch.no_grad():
        for (q, a, i) in zip(val_qs, val_ans, val_img):
            if torch.cuda.is_available():
                i = torch.Tensor(i).to(device)

            question_id = q[0]
            
            q = question_tokenizer(q[1], max_length=30, padding='max_length', return_tensors = 'pt')
            q = q.to(device)

            prediction = model(i, q, predict=True).to(device)
            predicted_idx = torch.argmax(prediction, dim=0)

            if classes[predicted_idx] == a:
                counter += 1

            results.append({"answer": classes[predicted_idx], "question_id": question_id})

    json.dump(results, open('results-concat.json', 'w'))
                
    return counter / len(val_qs)


def val_add(val_questions, val_answers, val_images, img_model, txt_model, final_model, classes):
    val_qs = []
    val_ans = []
    val_img = []
    results = []
    counter = 0

    img_model.eval()
    txt_model.eval()
    final_model.eval()

    question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for (q, a) in zip(val_questions['questions'], val_answers['annotations']):
        val_ans.append(a['multiple_choice_answer'].replace(' ', ''))
        val_qs.append((q['question_id'], q['question']))
        val_img.append(val_images[q['image_id']])

    with torch.no_grad():
        for (q, a, i) in zip(val_qs, val_ans, val_img):
            if torch.cuda.is_available():
                i = torch.Tensor(i).to(device)

            question_id = q[0]
            
            q = question_tokenizer(q[1], max_length=30, padding='max_length', return_tensors = 'pt')
            q = q.to(device)

            img_emb = img_model(i).to(device)
            question_emb = txt_model(q).to(device)
            emb = torch.add(img_emb, question_emb)

            prediction = final_model(emb).to(device)
            predicted_idx = torch.argmax(prediction, dim=1)

            if classes[predicted_idx] == a:
                counter += 1

            results.append({"answer": classes[predicted_idx], "question_id": question_id})

    json.dump(results, open('results-add-txt.json', 'w'))
                
    return counter / len(val_qs)


def prediction(i, q, img_model, txt_model, final_model, classes):
    question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if torch.cuda.is_available():
        i = torch.Tensor(i).to(device)
    
    q = question_tokenizer(q, max_length=30, padding='max_length', return_tensors = 'pt')
    q = q.to(device)

    img_emb = img_model(i).to(device)
    question_emb = txt_model(q).to(device)
    emb = torch.add(img_emb, question_emb)

    prediction = final_model(emb).to(device)
    predicted_idx = torch.argmax(prediction, dim=1)

    print(classes[predicted_idx])


if __name__ == '__main__':
    print('Loading training text')
    with open ('/common/users/kcm161/Assignment4-code/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json', 'r') as question_file:
        train_questions = json.load(question_file)
    with open ('/common/users/kcm161/Assignment4-code/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json', 'r') as answer_file:
        train_answers = json.load(answer_file)

    print('Loading training images')
    train_images = pickle.load(open('/common/users/kcm161/Assignment4-code/train.pickle', 'rb'))
    val_images = pickle.load(open('/common/users/kcm161/Assignment4-code/val.pickle', 'rb'))

    train_qs = []
    train_ans = []
    # image id, question, question id
    for q in train_questions['questions']:
        train_qs.append(q['question'])
    for a in train_answers['annotations']:
        train_ans.append(a['multiple_choice_answer'].replace(' ', ''))

    print('Obtained questions and annotations')

    sorted_answer_freqs = sorted(Counter(train_ans).items(), key=lambda x: -x[1])[:1000]
    class_ids = {}
    classes = [x[0] for x in sorted_answer_freqs]

    idx = 0
    for key, value in sorted_answer_freqs:
        class_ids[key] = np.zeros(1000)
        class_ids[key][idx] = 1
        idx += 1

    train_top_questions = []
    train_top_answers = []
    for (q, a) in zip(train_questions['questions'], train_answers['annotations']):
        if a['multiple_choice_answer'].replace(' ', '') in class_ids:
            train_top_answers.append(a['multiple_choice_answer'].replace(' ', ''))
            train_top_questions.append((q['image_id'], q['question']))

    print('Cleaned top 1000 answers')

    with open ('/common/users/kcm161/Assignment4-code/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as question_file:
        val_questions = json.load(question_file)
    with open ('/common/users/kcm161/Assignment4-code/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json', 'r') as answer_file:
        val_answers = json.load(answer_file)

    dataset = VQADataset(train_images, train_top_questions, train_top_answers, class_ids)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64)
    bert = BertModel.from_pretrained('bert-base-uncased')

    mode = 'ADD'
    predict = True

    if mode == 'ADD':
        img_model = VQAImageModel()
        txt_model = VQATextModel(bert)
        final_model = VQAFinalModel()
        for param in txt_model.bert.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(list(img_model.parameters()) + list(txt_model.parameters()) + list(final_model.parameters()), lr=1e-4)
    else:
        model = VQAModel(bert)
        for param in model.bert.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()

    if predict:
        if mode == 'ADD':
            img_model.load_state_dict(torch.load('img_model-add.pt'))
            final_model.load_state_dict(torch.load('final_model-add.pt'))
            txt_model.load_state_dict(torch.load('txt_model-add.pt'))
        else:
            model.load_state_dict(torch.load('model.pt'))

    if torch.cuda.is_available():
        criterion = criterion.to(device)
        if mode == 'ADD':
            final_model = final_model.to(device)
            txt_model = txt_model.to(device)
            img_model = img_model.to(device)
        else:
            model = model.to(device)

    if mode == 'ADD':
        train_add(loader, img_model, txt_model, final_model, criterion, optimizer, 20)
    else:
        train_concat(loader, model, criterion, optimizer, 50)

    if mode == 'ADD':
        accuracy = val_add(val_questions, val_answers, val_images, img_model, txt_model, final_model, classes)
        print('Accuracy:', accuracy)
        # Inference 1
        # prediction(val_images[69366], 'What is the man doing?', img_model, txt_model, final_model, classes)
        # Inference 2
        # prediction(val_images[237568], 'How many photos can you see?', img_model, txt_model, final_model, classes)
        # Inference 3
        # prediction(val_images[240301], 'Is it daylight in this picture?', img_model, txt_model, final_model, classes)
        # Inference 4
        # prediction(val_images[240301], 'Why is there a gap between the roof and wall?', img_model, txt_model, final_model, classes)
    else:
        accuracy = val_concat(val_questions, val_answers, val_images, model, classes)
        print('Accuracy:', accuracy)
    