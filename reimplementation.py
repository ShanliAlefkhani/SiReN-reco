# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:17:14 2021

@author: MIDaSLab
"""

# Data loader
import pandas as pd
import argparse
import time
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm






class Data_loader():
    def __init__(self,dataset,version):
        self.dataset=dataset; self.version=version
        if dataset=='ML-1M':
            self.sep='::'
            self.names=['userId','movieId','rating','timestemp'];
            
            self.path_for_whole='./ml-1m/ratings.dat'
            self.path_for_train='./ml-1m/train_1m%s.dat'%(version)
            self.path_for_test='./ml-1m/test_1m%s.dat'%(version)
            self.num_u=6040; self.num_v=3952;
            
        
        elif dataset=='amazon':
            self.path_for_whole='./amazon-book/amazon-books-enc.csv'
            self.path_for_train='./amazon-book/train_amazon%s.dat'%(version)
            self.path_for_test='./amazon-book/test_amazon%s.dat'%(version)
            self.num_u=35736; self.num_v=38121;

        elif dataset=='yelp':
            self.path_for_whole='./yelp/YELP_encoded.csv'
            self.path_for_train='./yelp/train_yelp%s.dat'%(version)
            self.path_for_test='./yelp/test_yelp%s.dat'%(version)
            self.num_u=41772; self.num_v=30037;
            
        elif dataset=='gowalla':
            self.path_for_train='./go/training.dat'
            self.path_for_test='./go/test.dat'
            self.num_u=29858; self.num_v=40981;
        
        else:
            raise NotImplementedError("incorrect dataset, you can choose the dataset in ('ML-100K','ML-1M','amazon','yelp')")
        
        

    def data_load(self):
        if self.dataset=='ML-1M':
            self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False,random_state=self.version)
            self.train = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
            self.test = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
                
        elif self.dataset=='amazon':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train=pd.read_csv(self.path_for_train,index_col=0)
            self.test=pd.read_csv(self.path_for_test,index_col=0)
            

        elif self.dataset=='yelp':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train=pd.read_csv(self.path_for_train,index_col=0)
            self.test=pd.read_csv(self.path_for_test,index_col=0)
        
        elif self.dataset=='gowalla':
            # self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train=pd.read_csv(self.path_for_train,index_col=0)
            self.test=pd.read_csv(self.path_for_test,index_col=0)
        
        # return self.train_set, self.test_set
    



class training_data(Dataset):
    def __init__(self, data_class, num_ng, offset):
        super(training_data, self).__init__()
        self.data = data_class
        self.num_ng = num_ng
        self.offset = offset
        
        self.tr = self.data.train.values; self.tr[:,0:-1]-=1 ; self.tr[:,-1] = np.sign(self.tr[:,-1]-offset)
        self.user_dict = dict() # (consumed items, ratings)
        self.user_neg_candi = dict() # negative sampling candidates
        whole_ = set(np.arange(self.data.num_v))
        for j in range(self.data.num_u):
            self.user_dict[j] = self.tr[self.tr[:,0]==j][:,1:]
            self.user_neg_candi[j] = whole_ - set(self.tr[self.tr[:,0]==j][:,1])
        
        
        
    
    def _uniform_ng_sample(self):
        self.quadruplet = []
        for u in [j for j in self.user_dict]:
            for i,sgn in self.user_dict[u]:
                for t in range(self.num_ng):
                    j = np.random.randint(0,self.data.num_v)
                    while j in self.user_dict[u][:,0]:
                        j = np.random.randint(0,self.data.num_v)
                    self.quadruplet.append([u,i,j,sgn])
        
    def _degree_ng_sample(self):
        pass
    
    
    def __len__(self):
        return len(self.data.train) * self.num_ng
    def __getitem__(self,idx):
        return self.quadruplet[idx][0], self.quadruplet[idx][1], self.quadruplet[idx][2], self.quadruplet[idx][3]  # u,i,j,sgn
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type = str,
                        default = 'ML-1M',
                        help = "Dataset"
                        )
    parser.add_argument('--version',
                        type = int,
                        default = 1,
                        help = "Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1024,
                        help = "Batch size"
                        )

    parser.add_argument('--dim',
                        type = int,
                        default = 64,
                        help = "Dimension"
                        )
    parser.add_argument('--lr',
                        type = float,
                        default = 5e-3,
                        help = "Learning rate"
                        )
    parser.add_argument('--offset',
                        type = float,
                        default = 3.5,
                        help = "Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type = int,
                        default = 40,
                        help = "The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type = int,
                        default = 4,
                        help = "The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type = int,
                        default = 2,
                        help = "The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type = int,
                        default = 1000,
                        help = "The number of epochs"
                        )
    parser.add_argument('--reg',
                        type = float,
                        default = 0.05,
                        help = "Regularization coefficient"
                        )
    args = parser.parse_args()
    
    
    
    print("Data loading...")
    st = time.time()
    data_class = Data_loader(args.dataset, args.version)
    threshold = round(args.offset) # to generate ground truth set
    data_class.data_load() # load whole dataset
    print("Loading complete! (loading time:%s)"%(time.time()-st))
    
    training_ = training_data(data_class,args.K,args.offset)
    
    
    
    training_._uniform_ng_sample()
    
    
    
    
    