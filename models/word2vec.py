import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_dim = 300):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(self.vocab_size, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.vocab_size)
        
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

        
    def forward(self, X):
        X = X.view(-1, self.vocab_size)

        emb = self.fc1(X)
        X = self.fc2(emb)
        X = F.log_softmax(X, dim = 1)
        return X, emb



class Word2Vec:
    def __init__(self, epochs = 10, batch_size = 64, learning_rate = 0.05):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.criterion =  nn.NLLLoss()

        self.model = NotImplemented

    def load_model(self, path):
        model_state = torch.load(path)
        self.vocab_size = model_state['fc1.weight'].shape[1]
        self.model = Word2VecModel(self.vocab_size)
        self.model.load_state_dict(model_state)
    
    def save_model(self, path = "../trained_models/"):
        try:
            import google.colab
            path = ""
        except:
            pass
        
        torch.save(self.model.state_dict(), path + "word2vec-" + str(datetime.date.today()) + ".pt")

    def fit(self, headlines):
        self.headlines = headlines
        self.__build_vocab()
        self.word_context_idx = self.headlines.apply(self.__get_word_context_pairs)\
            .explode().dropna().reset_index(drop = True)
        
        self.dataloader = DataLoader(self.word_context_idx, batch_size = self.batch_size,\
             collate_fn = self.__collate_fn, shuffle=True)

        self.model = Word2VecModel(self.vocab_size)
        self.model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), self.learning_rate, momentum=0.9)
        
        self.__train()
        
    def predict(self, headlines):
        if self.model is None:
            print("Fit or load the model first")

        self.model.eval()
        
        return headlines.apply(self.__headline_to_embeddings)
    
    def __headline_to_embeddings(self, headline):
        embeddings = []

        for word in headline:
          word_idx = self.word_to_idx[word]
          ohe = self.__one_hot_encode(word_idx)

          _, embedding = self.model(ohe.to(device))
          embedding = embedding.detach().squeeze().cpu().numpy()
          embeddings.append(embedding)
        
        return embeddings

    def __build_vocab(self):
        try:
            self.vocab = self.headlines.explode().unique()
        except:
            raise ValueError("Headlines are not tokenized or otherwise wrong data format")

        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word:idx for (idx, word) in enumerate(self.vocab)}

    def __get_word_context_pairs(self, words):
        word_context_pairs = []
        for i in range(0, len(words) - 1):
            for j in range(i + 1, len(words)):
                if i != j:
                    word = self.word_to_idx[words[i]]
                    context = self.word_to_idx[words[j]]
                    word_context_pairs.append((word, context))
                    word_context_pairs.append((context, word))
                
        return word_context_pairs
    
    def __one_hot_encode(self, word_idx):
        ohe = torch.zeros((self.vocab_size, 1), dtype = torch.float)
        ohe[word_idx] = 1

        return ohe
    
    def __collate_fn(self, word_context_pairs):
      words, contexts = [], []
      try:
        for word, context in word_context_pairs:
          word = self.__one_hot_encode(word)
          context = torch.tensor(context)
          words.append(word)
          contexts.append(context)
      except:
        print(word_context_pairs)
      
      return torch.stack(words), torch.stack(contexts)
  
    def __train(self):
        self.model.train()
        print("Training started")
        for epoch in range(self.epochs):
            for words, contexts in tqdm(self.dataloader):
                self.optimizer.zero_grad()

                pred_contexts, embs = self.model(words.to(device))
                pred_contexts = pred_contexts.squeeze()

                loss = self.criterion(pred_contexts.to(device), contexts.to(device))

                loss.backward()
                self.optimizer.step()
        print("Training finished")
        self.save_model()
        print("Model saved")
    

    