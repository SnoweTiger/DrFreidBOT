import random

# SK Learn import
# import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# PyTorch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

TRAIN_TEXT_FILE_PATH = 'phrases.txt'
PATH = 'DrFreid_NN_loss_1894.pt'
restricted_chars = 'qwertyuiop[]asdfghjkl;zxcvbnm,.?!'

BOT_CONFIG = {
  'intents': {
    'hello': {
      'question': ['категорически приветсвую', 'приветствую', 'привет', 'здраствуйте'],
      'response': ['Здравствуйте.','Привет.', 'Приветствую.', 'Привет, человек.',
                   'Здравствуй', 'Здравствуй, человек','Здравствуй, низшая форма жизни =)',
                   'Здравствуйте, рад вас видеть', 'Доброго времени суток.','Добрый день',
                    'привет человек']
      },
    'name':{
      'question': ['ты кто', 'какое твоё имя', 'как тебя величать', 'как к тебе обращаться','у тебя есть имя'],
      'response': ['Меня зовут  DrFreid, я бот.','Я бот обученый на изречения Зигизмунда Фрейда, можете меня нызщывать DrFreid']},
    'age':{
      'question': ['когда твой день рождения','когда ты родился','откуда ты родом','где ты родился'],
      'response': ['Я родился 6 мая 1856 года в небольшом городе Фрайберг в Моравии.  Сейчас моя родная улица носит мое имя.']},
    'whatcanyoudo': {
      'question': ['что ты умеешь?', 'расскажи что умеешь'],
      'response': ['Отвечать на вопросы. Просто напиши :)']},
    'whosyourdaddy': {
      'question': ['Кто твой создатель?', 'Кто тебя написал?', 'Кто тебя создал?', 'Кто твой автор?'],
      'response': ['И сказал программист: ""def go_bot()!"". И стал бот']}
    },
  'failure_phrases': ['Я не знаю, что ответить','Не поняла вас','Переформулируйте пожалуйста']}

dataset = []
for intent, intent_data in BOT_CONFIG['intents'].items():
    for question in intent_data['question']:
        question = question.lower()
        question = question.strip('?!')
        question = question.strip('')
        dataset.append([question, intent])

X_text = [x for x, y in dataset]
y = [y for x, y in dataset]
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Как улучшить?
X = vectorizer.fit_transform(X_text)
clf = LogisticRegression()
clf.fit(X, y)

# PyTorch model
class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key = lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
#     print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char

def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text

def get_gen_response(ask_text, model,char_to_idx,idx_to_char):
    response = evaluate(
        model,char_to_idx,idx_to_char,
        temp=0.3,prediction_len=100,start_text=ask_text
        )

    ask_text_len = len(ask_text)
    response_len = len(response)
    response_first_space = response.find(' ', ask_text_len)
    response_last_space =  response_len - response.rfind(' ')
    response_last_punctuation = response_len - max([
    response.rfind('.'),response.rfind(',')])

    if response_last_punctuation >= 3 and response_last_space >= 3:
        response = response[response_first_space:]
    else:
         response = response[response_first_space : (1-response_last_punctuation)]
    response = response.strip(' ')
    return response

def load_text(phrases_path):
    freid_phrases = []
    freid_text = ''

    with open(phrases_path, 'r', encoding="utf-8") as text_file:
        phrases_load = text_file.read()
        phrases_load = str(phrases_load)
        freid_phrases = phrases_load.split('\n\n')

    for  phrase in freid_phrases:
        phrase = phrase.replace(';', '')
        phrase = phrase.replace('—', '–')
        phrase = phrase.replace('(', '')
        phrase = phrase.replace(')', '')
        freid_text += ' ' + phrase

    return text_to_seq(freid_text)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

# Function SK model
def get_intent(text):
    probas = clf.predict_proba(vectorizer.transform([text]))[0]
    proba = max(probas)
#    print(proba)
    if proba > 0.5:
        index = list(probas).index(proba)
        return clf.classes_[index]

# Function for Bot
def bot_response(question, model,char_to_idx,idx_to_char):
    if not question:
        phrases = BOT_CONFIG['failure_phrases']
        return random.choice(phrases)
    intent = get_intent(question)
    if intent:
        phrases = BOT_CONFIG['intents'][intent]['response']
        return random.choice(phrases)
    return get_gen_response(question, model,char_to_idx,idx_to_char)

# Start code

sequence, char_to_idx, idx_to_char = load_text(TRAIN_TEXT_FILE_PATH)
device = torch.device('cpu')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
load_model(model, PATH)

while True:
    question = input('Вы: ')
    question = question.lower()
    question = question.strip(restricted_chars)
    question = question.strip('')
    if question == 'пока': break
    response_text = bot_response(question, model,char_to_idx,idx_to_char)
    print(f'Bot: {response_text}')
