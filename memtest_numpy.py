import numpy as np
import awkward as ak
import uproot
import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import timeit

def chunk_processing(file, _pad, max_num_particles, particle_features, jet_features, labels):
    chunk_X = []
    chunk_y = []
    for table in uproot.iterate(file, step_size=10000, library="np"):
        # print('table from rdf')
        # print(table)
        x_particles = np.stack([_pad(table[n], maxlen=max_num_particles) for n in particle_features], axis=1)
        events, part_ft, pad_size = np.shape(x_particles)
        x_particles = np.reshape(x_particles, (events, part_ft*pad_size))
        # x_particles.type.show()
        # print('flattened')
        x_jets = np.stack([table[n] for n in jet_features], axis=1)
        # x_jets.type.show()
        chunk_X.append(np.concatenate([x_particles, x_jets], axis=1))
        chunk_y.append(np.stack([table[n] for n in labels], axis=1))
    return chunk_X, chunk_y

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

def read_file(
        file_names,
        batch_size=128,
        max_num_particles=128,
        particle_features=['part_px', 'part_py', 'part_pz', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded. 
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """
    # names = ROOT.std.vector('string')()
    # for n in file_names: names.push_back(n) 
    # particle_features+jet_features+labels
    
    all_chunk_X = []
    all_chunk_y = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(chunk_processing, file, _pad, max_num_particles, particle_features, jet_features, labels ) for file in file_names]
        for future in tqdm(futures):
            (X, y) = future.result()
            all_chunk_X.extend(X)
            all_chunk_y.extend(y)

    return DataLoader(dataset=TensorDataset(
        torch.from_numpy(np.concatenate(all_chunk_X)), 
        torch.from_numpy(np.concatenate(all_chunk_y))),
        batch_size=batch_size, shuffle=True)

tree_name = "tree"
batch_size = 128
vec_columns=['part_px', 'part_py', 'part_pz', 'part_energy']
columns=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
targets = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']

print('Getting path of JetClass datasets ...')
import os
data_path = '/home/northnpk/Downloads/JetClass_dataset/'
train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]

# print(f'Train path: {train_path}')

train_loader = read_file(train_path[:10], batch_size=batch_size)
# Get a list of the columns used for training

num_features = int(train_loader.dataset[0][0].size()[0])
num_targets = int(train_loader.dataset[0][1].size()[0])

print(f'Number of input columns: {num_features}')
print(f'Number of targets columns: {num_targets}')

print(f'Initializing the models')
 
# Initialize PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(num_features, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, num_targets),
    torch.nn.Sigmoid(),
)

loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(model)

print('Start training...')
 # Loop through the training set and train model

def train():
    sum_loss = 0
    for i, (x_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Make prediction and calculate loss
        # tmp = (x_train, y_train)
        pred = model(x_train)
        # print(pred.to_numpy)
        loss = loss_fn(pred, y_train.to(torch.float32))
    
        # improve model
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Calculate accuracy
        sum_loss += loss
        
    print(f"Training => loss: {sum_loss/len(train_loader.dataset)}")
# import memray

# for e in range(10):
#     if e in [0,1,9]:
#         with memray.Tracker(f"awkwardtorch_epoch_{e}.bin", native_traces=True):
#             train()
#     else :
#         train()

print(f"Awkward Training time avg.: {timeit.timeit('train()', number=3, globals=globals())/3.0}")
print('Finished training')
print('Ended')