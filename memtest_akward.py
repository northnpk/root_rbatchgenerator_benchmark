import numpy as np
import awkward as ak
import uproot
import torch
import vector
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
# from ../ParticleTransformers import 
import timeit

def chunk_processing(file, _pad, max_num_particles, particle_features, jet_features, labels):
    chunk_x_part = []
    chunk_x_jet =[]
    chunk_x = []
    chunk_y = []
    for table in uproot.iterate(file, step_size=10000):
        p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
        table['part_pt'] = p4.pt
        table['part_eta'] = p4.eta
        table['part_phi'] = p4.phi

        x_particles = np.stack([_pad(table[n], maxlen=max_num_particles) for n in particle_features], axis=1)
        events, part_ft, pad_size = np.shape(x_particles)
        x_particles = np.reshape(x_particles, (events, part_ft*pad_size))
        # x_particles.type.show()
        # print('flattened')
        x_jets = np.stack([table[n] for n in jet_features], axis=1)
        # x_jets.type.show()
        # chunk_x_part.append(x_particles)
        # chunk_x_jet.append(x_jets)
        chunk_x.append(np.concatenate([x_particles, x_jets], axis=1))
        chunk_y.append(np.stack([table[n] for n in labels], axis=1))
    # return chunk_x_part, chunk_x_jet, chunk_y
    return chunk_x, chunk_y


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
        input_feature=['part_px', 'part_py', 'part_pz', 'part_energy',
                       'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy',
                       'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                       'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'],
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    # names = ROOT.std.vector('string')()
    # for n in file_names: names.push_back(n) 
    # particle_features+jet_features+labels
    
    all_chunk_x = []
    all_chunk_x_part = []
    all_chunk_x_jet = []
    all_chunk_y = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(chunk_processing, file, _pad, max_num_particles, particle_features, jet_features, labels ) for file in file_names]
        for future in tqdm(futures):
            # (x_part, x_jet, y) = future.result()
            # all_chunk_x_part.extend(x_part)
            # all_chunk_x_jet.extend(x_jet)
            # all_chunk_y.extend(y)
            (x, y) = future.result()
            all_chunk_x.extend(x)
            all_chunk_y.extend(y)

    # return DataLoader(dataset=TensorDataset(
    #     torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_part))), 
    #     torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_jet))), 
    #     torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_y)))),
    #     batch_size=batch_size, shuffle=True)
    return DataLoader(dataset=TensorDataset(
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x))), 
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_y)))),
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
class CIFAR10Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.3)
 
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = torch.nn.Flatten()
 
        self.fc3 = torch.nn.Linear(672, 512)
        self.act3 = torch.nn.ReLU()
        self.drop3 = torch.nn.Dropout(0.5)
 
        self.fc4 = torch.nn.Linear(512, 10)
 
    def forward(self, x):
        b, s = x.shape
        x = x.reshape(b, 4, 43, 3)
        # input 4x43x3
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x
 
model = CIFAR10Model()

loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(model)

print('Start training...')
 # Loop through the training set and train model

def train():
    sum_loss = 0
    for i, (x_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Make prediction and calculate loss        
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
# train()
# for e in range(10):
#     if e in [0,1,9]:
#         with memray.Tracker(f"awkwardtorch_epoch_{e}.bin", native_traces=True):
#             train()
#     else :
#         train()

print(f"Awkward Training time avg.: {timeit.timeit('train()', number=3, globals=globals())/3.0}")
print('Finished training')
print('Ended')