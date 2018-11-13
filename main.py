import sys
import os
import torch
import random
import numpy as np
import torch.optim as optim
import math
from util import cmd_args, load_data
import util_dinh as ud
from sklearn.model_selection import StratifiedKFold
from final_model import Classifier, loop_dataset
import shutil

if __name__ == '__main__':
    random.seed(cmd_args.seed)

    cur_dir = os.getcwd()
    statistics_dir = os.path.join(cur_dir, "statistics")
    if not os.path.exists(statistics_dir):
        os.makedirs(statistics_dir)
    save_dir = os.path.join(statistics_dir, cmd_args.data + "_" + cmd_args.gm +"_"+ str(cmd_args.learning_rate) + "_" + str(cmd_args.out_dim))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    else:
        os.makedirs(save_dir)
    model_dir = os.path.join(save_dir, "models")
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    results_dir = os.path.join(save_dir, "results")
    if not os.path.exists(results_dir):
       os.makedirs(results_dir)
    shuffle_dir = os.path.join(cur_dir, "shuffle_idx")
    
    graphs = load_data()

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    skf = StratifiedKFold(n_splits=10)

    for shuffle_idx in range(1,2):
        parameters_save = []
        random_idx = [int(idx) for idx in ud.load_list_from_file(shuffle_dir + '/' + cmd_args.data + "_" + str(shuffle_idx))]
        graphs_shuffle = [graphs[idx] for idx in random_idx]
        graphs=graphs_shuffle[:]
        labels = [g.label for g in graphs]
        
        fold_idx = 1
        for list_tr_idx, list_te_idx in skf.split(np.zeros(len(labels)), labels):            
            te_graphs = [graphs[idx] for idx in list_te_idx]
            tr_graphs = [graphs[idx] for idx in list_tr_idx[:-len(te_graphs)]]
            vali_graphs = [graphs[idx] for idx in list_tr_idx[-len(te_graphs):]]
            
            tr_idxes = list(range(len(tr_graphs)))
            vali_idxes = list(range(len(vali_graphs)))
            te_idxes = list(range(len(te_graphs)))
    
            best_model_path = model_dir + '/' + str(shuffle_idx) + "_" + str(fold_idx)
            classifier = Classifier()
            if cmd_args.mode == 'gpu':
                classifier = classifier.cuda()
    
            optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

            best_loss = None
            patience_count = 0                 
            for epoch in range(cmd_args.num_epochs):
                classifier.train()
                avg_loss,_ = loop_dataset(tr_graphs, classifier, tr_idxes, optimizer=optimizer)
                print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
    
                classifier.eval()
                vali_loss,_ = loop_dataset(vali_graphs, classifier, vali_idxes)                
                print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, vali_loss[0], vali_loss[1]))

                if epoch==0:
                    best_loss = vali_loss[0]
                    torch.save(classifier.state_dict(), best_model_path)
                    
                elif vali_loss[0] < best_loss:
                    torch.save(classifier.state_dict(), best_model_path)
                    best_loss = vali_loss[0]
                    patience_count = 0
                else:
                    patience_count+=1       
                
                if patience_count >= cmd_args.num_patience:
                    break

            optimal_model = Classifier()
            
            if cmd_args.mode == 'gpu':
                optimal_model = optimal_model.cuda()
            optimal_model.load_state_dict(torch.load(best_model_path))
            optimal_model.eval()
            test_loss, test_acc = loop_dataset(te_graphs, optimal_model, te_idxes)
            with open(results_dir + '/' + str(shuffle_idx), 'a+') as f:
                f.write(str(test_loss[1]) + '\n')
            fold_idx+=1
        print("--------------------------")
    print("=========================")
