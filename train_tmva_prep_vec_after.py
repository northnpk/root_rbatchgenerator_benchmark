# Import Library
import os
import timeit
# from memory_profiler import profile

import ROOT
from ROOT import RDataFrame

import vector
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

# Define Torch IterableDataset for TMVA RBatchGenerator
class RBatchDataset(IterableDataset):
    def __init__(self, generator, vector_prep_fn=None):
        self.generator = generator
        self.batch_size = self.generator.base_generator.batch_size
        self.vector_prep_fn = vector_prep_fn
        self.apply_vector_prep_fn = False
        if self.vector_prep_fn != None:
            self.apply_vector_prep_fn = True
        # print(f'Batch size: {self.batch_size}')
        if self.generator.last_batch_no_of_rows != 0 and self.generator.number_of_batches > 1:
            self.length = ((self.generator.number_of_batches-1) * self.batch_size) + self.generator.last_batch_no_of_rows
        elif self.generator.number_of_batches == 1:
            self.length = self.generator.last_batch_no_of_rows
        else :
            self.length = (self.generator.number_of_batches * self.batch_size)
        
        self.vec_columns = [c not in self.generator.base_generator.given_columns for c in self.generator.base_generator.train_columns]
        self._columns = [c in self.generator.base_generator.given_columns for c in self.generator.base_generator.train_columns]
        self.label_columns = self.generator.base_generator.target_columns
        self.vec_names = []
        for n in self.generator.base_generator.given_columns:
            if n not in self.label_columns and n not in self.generator.base_generator.train_columns:
                self.vec_names.append(n)
    
    def collate_fn(self, data):
        tensors, targets = data
        if self.apply_vector_prep_fn :
            return self.vector_prep_fn(tensors[:, self.vec_columns].reshape(len(tensors), 4, 128), self.vec_names), tensors[:, self._columns], targets
        
        else :
            return tensors[:, self.vec_columns].reshape(len(tensors), 4, 128), tensors[:, self._columns], targets
    
    def __iter__(self):
        return self.generator.__iter__()
        
    def __len__(self):
        return self.generator.number_of_batches

# array(torch->numpy)->numpy->torch
def vector_prep_fn_3(tensors, vec_names):
    tensors = tensors.numpy()
    p4 = vector.arr({name.replace('part_', ''): tensors[:, idx] for idx, name in enumerate(vec_names)})
    return torch.from_numpy(np.stack((p4.pt, p4.eta, p4.phi, tensors[:, -1]), axis=1))


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
                            'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']) -> DataLoader:
    # Passing vector columns
    max_vec_sizes = dict(zip(vec_columns, [max_num_particles]*len(vec_columns)))

    # get files path
    train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]

    # load RDataFrame from path
    r_df = RDataFrame(tree_name, train_path[:10])
    # Getting RBatchGenerator from tmva RBatchGenerator
    gen_train = ROOT.TMVA.Experimental.CreatePyTorchGenerators(r_df, 
                                                               batch_size, 
                                                               chunk_size, 
                                                               columns=columns+vec_columns+targets, 
                                                               max_vec_sizes=max_vec_sizes, 
                                                               target=targets, 
                                                               validation_split=0, 
                                                               drop_remainder=False)
    
    # load Generator to torch dataloader
    dataset = RBatchDataset(gen_train, vector_prep_fn=vector_prep_fn_3)
    return DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=None)
    
# Define Torch Model
class SimpleJetClassModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.part_conv = torch.nn.Conv1d(4, 1, 3, stride=1, padding= "same")
        self.part_pool = torch.nn.AvgPool1d(2, 2)
        self.jet_liner = torch.nn.Linear(4, 4)
        self.drop1 = torch.nn.Dropout(0.5)
        self.drop2 = torch.nn.Dropout(0.5)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten()
        self.fc = torch.nn.Linear(68, 10)
 
    def forward(self, x_part, x_jet):
        x_part = self.act1(self.part_conv(x_part))
        x_part = self.drop1(x_part)
        x_part = self.part_pool(x_part)
        x_part = self.flat(x_part)
        x_jet = self.act2(self.jet_liner(x_jet))
        x_jet = self.drop2(x_jet)
        return self.fc(torch.concat((x_part, x_jet), dim=1))

# @profile
# Define Training loop
def train(model, loss_fn, optimizer, train_loader):
    sum_loss = 0
    for (x_part, x_jet, y) in tqdm(train_loader):
        # Make prediction and calculate loss
        pred = model(x_part, x_jet)
        loss = loss_fn(pred, y.to(torch.float32))

        # improve model
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Calculate accuracy
        sum_loss += loss.item()*y.size(0)

    print(f"Training => loss: {sum_loss/train_loader.dataset.length}")

if __name__=="__main__":
    # Create model
    model = SimpleJetClassModel()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(model)

    # Create Dataloader
    print("Preparing Dataloader from TMVA")
    start = timeit.default_timer()
    train_loader = load_dataset()
    print("Done")

    # Train 1 Epoch
    print("Start training")
    train(model, loss_fn, optimizer, train_loader)
    print(f'RBatchGenerator time count {timeit.default_timer()-start}')
    print("Done")