# Import Library
import os
import timeit
# from memory_profiler import profile

import ROOT
from ROOT import RDataFrame
from ROOT import RDF

# import numpy as np

# from tqdm.auto import tqdm

# @profile
# Define load_dataset method to loading TMVA RBatchGenerator into Torch Dataloader
def load_dataset(data_path='/home/northnpk/Downloads/JetClass_dataset/',
                 tree_name = "tree",
                 batch_size = 128,
                 max_num_particles=128,
                 chunk_size = 5_000,
                 vec_columns=['part_px', 'part_py', 'part_pz', 'part_energy'],
                 columns=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
                 targets = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                            'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    # Passing vector columns
    max_vec_sizes = dict(zip(vec_columns, [max_num_particles]*len(vec_columns)))

    # get files path
    train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]

    # load RDataFrame from path
    r_df = RDataFrame(tree_name, train_path[:10])
    # Getting RBatchGenerator from tmva RBatchGenerator
    template = "float&,float&,float&,float&,ROOT::RVec<float>,ROOT::RVec<float>,ROOT::RVec<float>,ROOT::RVec<float>,float&,bool&,bool&,bool&,bool&,bool&,int&,int&,int&,int&"
    gen_train = ROOT.TMVA.Experimental.Internal.RBatchGenerator(template)(
            RDF.AsRNode(r_df),
            chunk_size,
            batch_size,
            columns+vec_columns+targets,
            526,
            [128, 128, 128, 128],
            0,
            0,
            0,
            True,
            False,
        )
    
    # load Generator to torch dataloader
    return gen_train.DeActivate()


# @profile
# Define Training loop
def train(model, loss_fn, optimizer, train_loader):
    sum_loss = 0
    for data in train_loader:
        x_part, y = data
        pass

if __name__=="__main__":
    
    # Create Dataloader
    print("Preparing Dataloader from TMVA")
    start = timeit.default_timer()
    train_loader = load_dataset()
    # print(timeit.timeit("load_dataset()", globals=globals()))
    # print("Done")

    # Train 1 Epoch
    # print("Start training")
    # train(None, None, None, train_loader)
    print(f'RBatchGenerator time count {timeit.default_timer()-start}')
    print("Done")