# Import Library
import os
# import timeit
from memory_profiler import profile

import numpy as np
import awkward as ak
import vector
import uproot

import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from torchinfo import summary

# Define padding method to pad vector columns
def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x
    
# Define Chunk Processing with Uproot
def chunk_processing(file, max_num_particles, particle_features, jet_features, labels):
    chunk_x_part = []
    chunk_x_jet =[]
    chunk_y = []
    for table in uproot.iterate(file, step_size=5000):
        p4 = vector.zip({'px': table['part_px'],
                         'py': table['part_py'],
                         'pz': table['part_pz'],
                         'energy': table['part_energy']})
        table['part_pt'] = p4.pt
        table['part_eta'] = p4.eta
        table['part_phi'] = p4.phi

        x_particles = np.stack([_pad(table[n], maxlen=max_num_particles) for n in particle_features], axis=1)
        x_jets = np.stack([table[n] for n in jet_features], axis=1)
        y = np.stack([table[n] for n in labels], axis=1)

        chunk_x_part.append(x_particles)
        chunk_x_jet.append(x_jets)
        chunk_y.append(y)
    return chunk_x_part, chunk_x_jet, chunk_y

@profile
# Define load_dataset method to read multiple root files with chunk_processing and load into Torch Dataloader
def load_dataset(data_path='/home/northnpk/Downloads/JetClass_dataset/',
                 batch_size=128,
                 max_num_particles=128,
                 particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
                 jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
                 labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                         'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']) -> DataLoader:
    # Getting files path
    train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]
    
    all_chunk_x_part = []
    all_chunk_x_jet = []
    all_chunk_y = []
    # Multi-Processor to get all chunk of dataset
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(chunk_processing, file, max_num_particles, particle_features, jet_features, labels ) for file in train_path[:10]]
        for future in futures:
            (x_part, x_jet, y) = future.result()
            all_chunk_x_part.extend(x_part)
            all_chunk_x_jet.extend(x_jet)
            all_chunk_y.extend(y)

    return DataLoader(dataset=TensorDataset(
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_part))), 
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_jet))), 
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_y)))),
        batch_size=batch_size, shuffle=True)
    
# Define Torch Model
class ParticleFlowNetwork(torch.nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 Phi_sizes=(100, 100, 128),
                 F_sizes=(100, 100, 100),
                 use_bn=True,
                 for_inference=False,
                 **kwargs):

        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = torch.nn.BatchNorm1d(input_dims) if use_bn else torch.nn.Identity()
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(torch.nn.Sequential(
                torch.nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1),
                torch.nn.BatchNorm1d(Phi_sizes[i]) if use_bn else torch.nn.Identity(),
                torch.nn.ReLU())
            )
        self.phi = torch.nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(torch.nn.Sequential(
                torch.nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                torch.nn.ReLU())
            )
        f_layers.append(torch.nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(torch.nn.Softmax(dim=1))
        self.fc = torch.nn.Sequential(*f_layers)

    def forward(self, features, x_jets):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        x = self.input_bn(features)
        x = self.phi(x)
        x = x.sum(-1)
        return self.fc(x)

@profile
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

    print(f"Training => loss: {sum_loss/len(train_loader.dataset)}")


if __name__=="__main__":
    # Create model
    Phi_sizes = (128, 128, 128)
    F_sizes = (128, 128, 128)
    input_dims = 4
    num_classes = 10
    model = ParticleFlowNetwork(input_dims, num_classes, Phi_sizes=Phi_sizes,
                                F_sizes=F_sizes, use_bn=False)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(model)
    summary(model, [(1, 4, 128,), (1, 4,)], device='cpu')

    # Create Dataloader
    print("Preparing Dataloader from Uproot")
    # start = timeit.default_timer()
    train_loader = load_dataset()
    print("Done")

    # Train 1 Epoch
    print("Start training")
    train(model, loss_fn, optimizer, train_loader)
    # print(f'Awkward+Uproot time count {timeit.default_timer()-start}')
    print("Done")