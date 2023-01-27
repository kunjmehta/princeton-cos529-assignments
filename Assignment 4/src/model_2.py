import json
import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import re
import spacy
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def __init__(self, vocab_size, glove_weights, embedding_dim=300, hidden_dim=512, num_classes=1000) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.dropout = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, hidden_dim))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        
    def forward(self, img, text):
        text_emb = self.embeddings(text)
        text_emb = self.dropout(text_emb)
        if len(text_emb.shape) == 2:
            text_emb = torch.unsqueeze(text_emb, 0)
        lstm_out, (ht, ct) = self.lstm1(text_emb)
        lstm_out, (ht, ct) = self.lstm2(lstm_out)

        img_emb = self.fc1(img)
        if len(img_emb.shape) == 1:
            img_emb = torch.unsqueeze(img_emb, 0)

        emb = img_emb + ht[-1]
        emb = self.fc2(emb)
        return emb


def train(loader, model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print('Starting epoch {}'.format(epoch))
        model.train()
        running_loss = 0.
        last_loss = 0.

        for i, (img, question, label) in enumerate(loader):
            if torch.cuda.is_available():
                img, question, label = img.to(device), question.to(device), label.to(device)

            prediction = model(img, question).to(device)
            predicted_idx = torch.argmax(prediction, dim=1)
            label_idx = torch.argmax(label, dim=1)
            # print(predicted_idx, label_idx)
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

    torch.save(model.state_dict(), 'model2.pt')


def val(val_questions, val_answers, val_images, model, classes):
    val_qs = []
    val_ans = []
    val_img = []
    results = []
    counter = 0

    model.eval()

    for (q, a) in zip(val_questions, val_answers['annotations']):
        val_ans.append(a['multiple_choice_answer'].replace(' ', ''))
        val_qs.append((q[0], q[1], q[2]))
        val_img.append(val_images[q[0]])

    with torch.no_grad():
        for (q, a, i) in zip(val_qs, val_ans, val_img):
            question_id = q[2]
            if torch.cuda.is_available():
                i = torch.Tensor(i).to(device)
                q = torch.Tensor(q[1]).to(device).long()

            prediction = model(i, q).to(device)
            predicted_idx = torch.argmax(prediction, dim=1)

            if classes[predicted_idx] == a:
                counter += 1
            
            results.append({"answer": classes[predicted_idx], "question_id": question_id})

    json.dump(results, open('results_baseline-txt.json', 'w'))
                
    return counter / len(val_qs)


def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def encode_sentence(text, vocab2index, N=40):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded

def load_glove_vectors(glove_file="../glove.6B/glove.6B.100d.txt"):
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 100):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx

if __name__ == '__main__':
    with open ('/common/users/kcm161/Assignment4-code/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json', 'r') as question_file:
        train_questions = json.load(question_file)
    with open ('/common/users/kcm161/Assignment4-code/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json', 'r') as answer_file:
        train_answers = json.load(answer_file)

    train_images = pickle.load(open('/common/users/kcm161/Assignment4-code/train.pickle', 'rb'))
    val_images = pickle.load(open('/common/users/kcm161/Assignment4-code/val.pickle', 'rb'))

    train_qs = []
    train_ans = []
    # image id, question, question id
    for q in train_questions['questions']:
        train_qs.append(q['question'])
    for a in train_answers['annotations']:
        train_ans.append(a['multiple_choice_answer'].replace(' ', ''))

    sorted_answer_freqs = sorted(Counter(train_ans).items(), key=lambda x: -x[1])[:1000]
    class_ids = {}
    classes = [x[0] for x in sorted_answer_freqs]
    class_frequencies = [x[1] for x in sorted_answer_freqs]
    max_freq = class_frequencies[0]
    class_weights = [1 / (x / sum(class_frequencies)) for x in class_frequencies]

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

    tok = spacy.load('en_core_web_sm')
    counts = Counter()
    for question in train_top_questions:
        counts.update(tokenize(question[1]))

    print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:",len(counts.keys()))

    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    encoded_questions = []
    for question in train_top_questions:
        encoded_questions.append((question[0], encode_sentence(question[1], vocab2index)))
    encoded_questions = np.array(encoded_questions)

    with open ('/common/users/kcm161/Assignment4-code/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as question_file:
        val_questions = json.load(question_file)
    with open ('/common/users/kcm161/Assignment4-code/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json', 'r') as answer_file:
        val_answers = json.load(answer_file)

    val_encoded_questions = []
    for question in val_questions['questions']:
        val_encoded_questions.append((question['image_id'], encode_sentence(question['question'], vocab2index), question['question_id']))
    val_encoded_questions = np.array(val_encoded_questions)

    vocab_size = len(words)
    dataset = VQADataset(train_images, encoded_questions, train_top_answers, class_ids)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64)

    word_vecs = load_glove_vectors()
    pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)

    model = VQAModel(vocab_size, pretrained_weights, 100, 512)
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load('model2.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion = criterion.to(device)
        model = model.to(device)

    # train(loader, model, criterion, optimizer, 50)
    accuracy = val(val_encoded_questions, val_answers, val_images, model, classes)
    print('Accuracy:', accuracy)
