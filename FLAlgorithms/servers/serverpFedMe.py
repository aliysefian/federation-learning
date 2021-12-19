import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Implementation for pFedMe Server


def read_my_data(id):

    df= pd.read_csv("pre.csv")
    dt=df[(df.client_id == id)].to_numpy()
    ta,test=dt[:,:-2],dt[:,-2]
    X_train, X_test, y_train, y_test = train_test_split(dt[:,:-3],dt[:,-2], test_size=0.20)
    print("train",X_train.shape,"test:",X_test.shape,y_train.shape, y_test.shape)
    X_train = torch.Tensor(X_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

    
class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        # data = read_my_data()
        # data = read_data(dataset)

        total_users = 286
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):

            # id, train , test = read_user_data(i, data, dataset)
            id, train , test = read_my_data(i)

            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()


        #print(loss)
        self.save_results()
        self.save_model()
    
  
