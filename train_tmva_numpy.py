# Import Library
import os
import timeit
# from memory_profiler import profile

import ROOT
from ROOT import RDataFrame

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
                # vec_columns=['part_px', 'part_py', 'part_pz', 'part_energy'],
                 columns=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
                 targets = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                            'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    # Passing vector columns
    max_vec_sizes = dict(zip(vec_columns, [max_num_particles]*len(vec_columns)))

    # get files path
    train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]
    # train_path = ['/home/northnpk/Downloads/JetClass_dataset/train/TTBar_067.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW4Q_051.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToBB_079.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW4Q_021.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBarLep_054.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToBB_097.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBar_024.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW2Q1L_089.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBarLep_078.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToCC_035.root']

    # load RDataFrame from path
    # start = timeit.default_timer()
    num_files = 10
    print(f'Number of files: {num_files}')
    r_df = RDataFrame(tree_name, train_path[:num_files])
    # print(f'RDF initialize time count {timeit.default_timer()-start}')
    # Getting RBatchGenerator from tmva RBatchGenerator
    gen_train = ROOT.TMVA.Experimental.CreateNumPyGenerators(r_df, 
                                                               batch_size, 
                                                               chunk_size,
                                                               columns=vec_columns,
                                                            #    columns=columns+vec_columns+targets, 
                                                               max_vec_sizes=max_vec_sizes, 
                                                            #    target=targets, 
                                                               validation_split=0, 
                                                               drop_remainder=False)
    
    # load Generator to torch dataloader
    return gen_train


# @profile
# Define Training loop
def train(model, loss_fn, optimizer, train_loader):
    for data in train_loader:
        # x_part, y = data
        # print(x_part, y)
        print(data)
        # pass

if __name__=="__main__":
    
    # Create Dataloader
    print("Preparing Dataloader from TMVA")
    start = timeit.default_timer()
    train_loader = load_dataset()
    print(f'CreateNumPyGenerators initialize time count {timeit.default_timer()-start}')
    # print(timeit.timeit("load_dataset()", globals=globals()))
    # print("Done")

    # Train 1 Epoch
    print("Start training")
    start = timeit.default_timer()
    # train(None, None, None, train_loader)
    epoch = 1
    for _ in range(epoch): train(None, None, None, train_loader)
    print(f'RBatchGenerator iteration {epoch} epoch time count {timeit.default_timer()-start}')
    print("Done")