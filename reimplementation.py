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
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch import Tensor, nn, optim
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from torch_geometric.data import Data
from copy import deepcopy
import torch.nn.functional as F


class LightGConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self,x,edge_index):
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
    def update(self,inputs: Tensor) -> Tensor:
        return inputs
    
class SiReN(nn.Module):
    def __init__(self, data_class, args, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SiReN,self).__init__()
        self.data = data_class
        self.args = args
        
        
        edge_user = torch.tensor(self.data.train[self.data.train['rating']>self.args.offset]['userId'].values-1)
        edge_item = torch.tensor(self.data.train[self.data.train['rating']>self.args.offset]['movieId'].values-1) + self.data.num_u
        edge_ = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)
        self.data_p = Data(edge_index = edge_).to(device)
        self.__init_weight()
        self.get_normed_adj()
    def get_normed_adj(self):
        num_nodes = self.data.num_u + self.data.num_v
        indices, values = get_laplacian(self.data_p.edge_index, normalization = "sym", num_nodes = num_nodes)
        iden = torch.vstack((torch.arange(0,num_nodes),torch.arange(0,num_nodes)))
        values_iden = torch.ones(num_nodes)
        if torch.cuda.is_available():
            I = torch.sparse.FloatTensor(iden,values_iden,size = (num_nodes,num_nodes)).cuda()
        else:
            I = torch.sparse.FloatTensor(iden,values_iden,size = (num_nodes,num_nodes))
        self.L = torch.sparse.FloatTensor(indices,-values,size=(num_nodes,num_nodes)) + I
        
        
        
    
    
    def __init_weight(self):
        # generate embeddings
        self.embeddings_pos = nn.Embedding(num_embeddings = self.data.num_u + self.data.num_v , embedding_dim = self.args.dim)
        self.embeddings_neg = nn.Embedding(num_embeddings = self.data.num_u + self.data.num_v , embedding_dim = self.args.dim)
        
        # convs for pos
        self.convs = nn.ModuleList()
        for _ in range(self.args.num_layers):
            self.convs.append(LightGConv())
        
        # mlps for neg
        self.mlps = nn.ModuleList()
        for _ in range(self.args.MLP_layers):
            self.mlps.append(nn.Linear(self.args.dim,self.args.dim,bias = True))
        
        # Attention model
        self.attn = nn.Linear(self.args.dim,self.args.dim,bias=True)
        self.q = nn.Linear(self.args.dim, 1 , bias=False)
        self.attn_softmax = nn.Softmax(dim=1)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.embeddings_pos.weight)
        nn.init.xavier_uniform_(self.embeddings_neg.weight)
        for _ in range(self.args.MLP_layers): 
            nn.init.xavier_uniform_(self.mlps[_].weight)
            nn.init.constant_(self.mlps[_].bias,0)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.constant_(self.attn.bias,0)
        nn.init.xavier_uniform_(self.q.weight)
        
        
    def aggregate(self):
        
        # positive graph
        POS = [self.embeddings_pos.weight]
        x = self.embeddings_pos.weight
        for i in range(1,self.args.num_layers):
            x = self.L.mm(x)
            POS.append(x)
        # torch-geometric
        # x = self.convs[0](self.embeddings_pos.weight,self.data_p.edge_index)
        # POS.append(x)
        # for i in range(1,self.args.num_layers):
        #     x = self.convs[i](x,self.data_p.edge_index)
        #     POS.append(x)
        
        # z_p = sum(POS)/len(POS)
        z_p = torch.stack(POS,dim=1)
        z_p = torch.mean(z_p,dim=1)
        # emb_u, emb_v = torch.split(z_p,[self.data.num_u,self.data.num_v])
        
    #     return emb_u, emb_v
    # def temp(self):
        # negative graph
        y = F.relu(self.mlps[0](self.embeddings_neg.weight))
        for i in range(1, self.args.MLP_layers):
            y = self.mlps[i](y)
            y = F.relu(y)
        y = F.dropout(y,p=0.5,training=self.training)
        z_n = y
        
        # Combine
        w_p = F.dropout(self.q(torch.tanh(self.attn(z_p))),p=0.5,training=self.training)
        w_n = F.dropout(self.q(torch.tanh(self.attn(z_n))),p=0.5,training=self.training)
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))
        
        Z = alpha_[:,0].view(len(z_p),1) * z_p + alpha_[:,1].view(len(z_p),1) * z_n
        
        emb_u, emb_v = torch.split(Z,[self.data.num_u,self.data.num_v])
        
        return emb_u, emb_v
    def forward(self,u,i,j,sgn,gamma = 1e-10):
        emb_u, emb_v = self.aggregate()
        u_ = emb_u[u]
        i_ = emb_v[i]
        j_ = emb_v[j]
        sgn = sgn.to("cuda:0" if torch.cuda.is_available() else "cpu")
        pos_scores = torch.mul(u_, i_).sum(dim=1)
        neg_scores = torch.mul(u_,j_).sum(dim=1)
        sBPRloss = -torch.log(gamma + torch.sigmoid(sgn * pos_scores - neg_scores)).mean()
        
        reg_loss = self.args.reg * (u_**2 + i_**2 + j_**2).sum(dim=-1).mean()
        
        return sBPRloss + reg_loss



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
        print("Prepare training dataset...")
        self.data = data_class
        self.num_ng = num_ng
        self.offset = offset
        
        self.tr = deepcopy(self.data.train.values); self.tr[:,0:-1]-=1 ; self.tr[:,-1] = np.sign(self.tr[:,-1]-offset)
        self.user_dict = dict() # (consumed items, ratings)
        # self.user_neg_candi = dict() # negative sampling candidates
        # whole_ = set(np.arange(self.data.num_v))
        for j in range(self.data.num_u):
            self.user_dict[j] = self.tr[self.tr[:,0]==j][:,1:]
            # self.user_neg_candi[j] = whole_ - set(self.tr[self.tr[:,0]==j][:,1])
        
        
        
    
    def _uniform_ng_sample(self):
        self.quadruplet = []
        pbar = tqdm(desc = "negative sampling ::",total = len([j for j in self.user_dict]),position=0)
        for u in [j for j in self.user_dict]:
            for i,sgn in self.user_dict[u]:
                for t in range(self.num_ng):
                    j = np.random.randint(0,self.data.num_v)
                    while j in self.user_dict[u][:,0]:
                        j = np.random.randint(0,self.data.num_v)
                    self.quadruplet.append([u,i,j,sgn])
            pbar.update(1)
        pbar.close()
        
    def _degree_ng_sample(self):
        pass
    
    
    def __len__(self):
        return len(self.data.train) * self.num_ng
    def __getitem__(self,idx):
        return self.quadruplet[idx][0], self.quadruplet[idx][1], self.quadruplet[idx][2], self.quadruplet[idx][3]  # u,i,j,sgn
    



def gen_top_k(data_class, r_hat, K=300):
    
    no_item = (torch.tensor(list(set(np.arange(1,data_class.num_v+1)) - set(data_class.train['movieId']))) - 1).long()
    r_hat[:,no_item] = -np.inf
    for u,i in data_class.train.values[:,:-1]-1:
        r_hat[u,i] = -np.inf
    reco = torch.topk(r_hat,K).indices.numpy()
    return reco
    


class evaluator():
    def __init__(self,data_class,reco,args,N=[10,15,20],partition=[20,50]):
        print('*** evaluation phase ***')
        
        self.reco = reco
        self.data = data_class
        self.N = np.array(N)
        self.threshold = round(args.offset) # to generate ground truth set
        self.partition = partition
        self.no_item = (np.array(list(set(np.arange(1,data_class.num_v+1)) - set(data_class.train['movieId']))) - 1)
        
        self.__gen_ground_truth_set()
        self.__group_partition()
        
        
    
    def __gen_ground_truth_set(self):
        print('*** ground truth set ***')
        self.GT = dict()
        temp = deepcopy(self.data.test); temp = temp[temp['rating']>=self.threshold].values[:,:-1]-1;
        for j in range(self.data.num_u):
            # if len(temp[temp[:,0]==j][:,1]>0) :  self.GT[j] = temp[temp[:,0]==j][:,1]
            if len(np.setdiff1d(temp[temp[:,0]==j][:,1],self.no_item))>0 :  self.GT[j] = np.setdiff1d(temp[temp[:,0]==j][:,1],self.no_item)
    
    def __group_partition(self):
        print('*** ground partition ***')
        unique_u, counts_u = np.unique(self.data.train['userId'].values-1, return_counts=True)
        self.G = dict()
        self.G['group1'] = unique_u[np.argwhere(counts_u<self.partition[0])].reshape(-1)
        temp = unique_u[np.argwhere(counts_u<self.partition[1])]
        self.G['group2'] = np.setdiff1d(temp,self.G['group1'])
        self.G['group3'] = np.setdiff1d(unique_u,temp)
        self.G['total'] = unique_u
        
    def precision_and_recall(self):
        print('*** precision ***')
        self.p = dict(); self.r = dict(); leng = dict()
        maxn = max(self.N)
        for i in [j for j in self.G]:
            self.p[i] = np.zeros(maxn); self.r[i] = np.zeros(maxn)
            leng[i] = 0
        for uid in [j for j in self.GT]:
            leng['total']+=1
            hit_ = np.cumsum([1.0 if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            self.p['total']+=hit_/ np.arange(1,maxn+1); self.r['total']+=hit_/len(self.GT[uid])
            if uid in self.G['group1']:
                self.p['group1']+=hit_/ np.arange(1,maxn+1); self.r['group1']+=hit_/len(self.GT[uid])
                leng['group1']+=1
            elif uid in self.G['group2']:
                self.p['group2']+=hit_/ np.arange(1,maxn+1); self.r['group2']+=hit_/len(self.GT[uid])
                leng['group2']+=1
            elif uid in self.G['group3']:
                self.p['group3']+=hit_/ np.arange(1,maxn+1); self.r['group3']+=hit_/len(self.GT[uid])
                leng['group3']+=1
        for i in [j for j in self.G]:
            self.p[i]/=leng[i]; self.r[i]/=leng[i]; 
            
    def normalized_DCG(self):
        print('*** nDCG ***')
        self.nDCG = dict(); leng=dict()
        maxn = max(self.N)
        
        for i in [j for j in self.G]:
            self.nDCG[i] = np.zeros(maxn)
            leng[i] = 0
        for uid in [j for j in self.GT]:
            leng['total']+=1
            idcg_len = min(len(self.GT[uid]),maxn)
            temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2,maxn+2)))
            temp_idcg[idcg_len:]=temp_idcg[idcg_len-1]
            temp_dcg = np.cumsum([1.0/np.log2(idx+2) if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            self.nDCG['total']+=temp_dcg / temp_idcg
            if uid in self.G['group1']:
                self.nDCG['group1']+=temp_dcg / temp_idcg
                leng['group1']+=1
            elif uid in self.G['group2']:
                self.nDCG['group2']+=temp_dcg / temp_idcg
                leng['group2']+=1
            elif uid in self.G['group3']:
                self.nDCG['group3']+=temp_dcg / temp_idcg
                leng['group3']+=1
        for i in [j for j in self.G]:
            self.nDCG[i]/=leng[i];
    







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
                        default = 1e-3,
                        help = "Learning rate"
                        )
    parser.add_argument('--offset',
                        type = float,
                        default = 3.5,
                        help = "Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type = int,
                        default = 5,
                        help = "The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type = int,
                        default = 2,
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
                        default = 5e-5,
                        help = "Regularization coefficient"
                        )
    args = parser.parse_args()
    
    
    
    print("Data loading...")
    st = time.time()
    data_class = Data_loader(args.dataset, args.version)
    threshold = round(args.offset) # to generate ground truth set
    data_class.data_load() # load whole dataset
    print("Loading complete! (loading time:%s)"%(time.time()-st))
    print('dataset :: %s with version %s'%(args.dataset,args.version))
    # data_class.train = data_class.train[data_class.train['rating']>3.5] # LightGCN
    
    
    # model preparation
    model = SiReN(data_class,args)
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    print("\nTraining on {}...\n".format("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.train()
    
    # dataset preparation
    training_ = training_data(data_class,args.K,args.offset)
    for EPOCH in range(1,args.epoch-1):
        training_._uniform_ng_sample()
        
        LOSS = 0
        ds = DataLoader(training_,batch_size = args.batch_size * args.K, shuffle=True)
        pbar = tqdm(desc = 'Version {} :: Epoch {}/{}'.format(args.version,EPOCH,args.epoch),total=len(ds),position=0)
        for batch_idx, (u,i,j,sgn) in enumerate(ds):
            optimizer.zero_grad()
            loss = model(u,i,j,sgn)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix({'loss':loss.item()})
        pbar.close()
            
        
        if EPOCH%1 ==0:
            model.eval()
            emb_u, emb_v = model.aggregate()
            emb_u = emb_u.cpu().detach(); emb_v = emb_v.cpu().detach()
            r_hat = emb_u.mm(emb_v.T)
            reco = gen_top_k(data_class,r_hat)
            eval_ = evaluator(data_class,reco,args)
            eval_.precision_and_recall()
            eval_.normalized_DCG()
            print("\n***************************************************************************************")
            print(" /* Recommendation Accuracy */")
            print('N :: %s'%(eval_.N))
            print('Precision at [10, 15, 20] :: ',eval_.p['total'][eval_.N-1])
            print('Recall at [10, 15, 20] :: ',eval_.r['total'][eval_.N-1])
            print('nDCG at [10, 15, 20] :: ',eval_.nDCG['total'][eval_.N-1])
            print("***************************************************************************************")
            model.train()
                        
        
    
    
    
    
    
    
    