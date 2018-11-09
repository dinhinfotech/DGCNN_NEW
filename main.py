import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from DGCNN_deepsets_embedding import DGCNNDeepSets

from mlp_dropout import MLPClassifier
sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from util import cmd_args, load_data
import util_dinh as ud
from sklearn.model_selection import StratifiedKFold, cross_val_score

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        elif cmd_args.gm == 'DGCNNDS':
            model = DGCNNDeepSets
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN' or cmd_args.gm == 'DGCNNDS':
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=0,
                            k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim,
                            num_edge_feats=0,
                            max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN' or cmd_args.gm == 'DGCNNDS' :
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)

        return self.mlp(embed, labels)

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
        #pbar.set_description((loss, acc))

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss, acc


if __name__ == '__main__':
    random.seed(cmd_args.seed)

    list_learning_rate = [0.001, 0.0001]
    n_fold = 10
    shuffle_idx_folder = "/shuffle_idx/"
    model_folder = "/models/"
    parameters_folder = "/parameters/"
    results_folder = "/results/"
    
    graphs = load_data()

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    skf = StratifiedKFold(n_splits=n_fold)

    list_accs = []
    for shuffle_idx in range(1,2):
        parameters_save = []
        random_idx = [int(idx) for idx in ud.load_list_from_file(os.getcwd() + shuffle_idx_folder + cmd_args.data + "_" + str(shuffle_idx))]
        graphs_shuffle = [graphs[idx] for idx in random_idx]
        labels = [g.label for g in graphs]
        
        fold_idx = 1
        for list_tr_idx, list_te_idx in skf.split(np.zeros(len(labels)), labels):            
            te_graphs = [graphs[idx] for idx in list_te_idx]
            tr_graphs = [graphs[idx] for idx in list_tr_idx[:-len(te_graphs)]]
            vali_graphs = [graphs[idx] for idx in list_tr_idx[-len(te_graphs):]]
            
            tr_idxes = list(range(len(tr_graphs)))
            vali_idxes = list(range(len(vali_graphs)))
            te_idxes = list(range(len(te_graphs)))
            dict_lr_loss = {}
    
            for lr_idx, lr in enumerate(list_learning_rate):
                best_model_path = os.getcwd() + model_folder + cmd_args.data + "_" + str(fold_idx) + "_" + str(lr_idx)
                classifier = Classifier()
                if cmd_args.mode == 'gpu':
                    classifier = classifier.cuda()
        
                optimizer = optim.Adam(classifier.parameters(), lr=lr)

                best_loss = None
                patience_count = 0                 
                for epoch in range(cmd_args.num_epochs):
                    classifier.train()
                    avg_loss, _ = loop_dataset(tr_graphs, classifier, tr_idxes, optimizer=optimizer)
                    #print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
        
                    classifier.eval()
                    vali_loss, vali_acc = loop_dataset(vali_graphs, classifier, vali_idxes)
                    #print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, validation_loss[0], validation_loss[1]))
                    
                    if epoch==0:
                        best_loss = vali_loss[0]
                        torch.save(classifier.state_dict(), best_model_path)
                        
                    if vali_loss[0] < best_loss:
                        torch.save(classifier.state_dict(), best_model_path)
                        best_loss = vali_loss[0]
                        patience_count = 0
                    else:
                        patience_count+=1       
                    
                    if patience_count >= cmd_args.num_patience:
                        break
                dict_lr_loss[lr_idx] = best_loss
            
            opt_lr_idx = min(dict_lr_loss, key=dict_lr_loss.get)
            opt_lr = list_learning_rate[opt_lr_idx]
            optimal_model_path = os.getcwd() + model_folder + cmd_args.data + "_" + str(fold_idx) + "_" + str(opt_lr_idx)         

            optimal_model = Classifier()
            
            if cmd_args.mode == 'gpu':
                optimal_model = optimal_model.cuda()
            optimal_model.load_state_dict(torch.load(optimal_model_path))
            optimal_model.eval()
            test_loss, test_acc = loop_dataset(te_graphs, optimal_model, te_idxes)
                        
            list_accs.append(test_loss[1])
            

            with open(os.getcwd() + parameters_folder + str(shuffle_idx) , 'a+') as f:
                f.write(str(fold_idx) + ", " + str(opt_lr) + ", " + str(dict_lr_loss[opt_lr_idx]) + '\n')            

            with open(os.getcwd()+ results_folder + str(shuffle_idx), 'a+') as f:
                f.write(str(test_loss[1]) + '\n')
            fold_idx+=1
        print("--------------------------")
    print("=========================")
