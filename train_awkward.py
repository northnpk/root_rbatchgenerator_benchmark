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
# from concurrent.futures import ProcessPoolExecutor
# from torchinfo import summary

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
    # train_path = ['/home/northnpk/Downloads/JetClass_dataset/train/TTBar_067.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW4Q_051.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToBB_079.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW4Q_021.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBarLep_054.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToBB_097.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBar_024.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToWW2Q1L_089.root', '/home/northnpk/Downloads/JetClass_dataset/train/TTBarLep_078.root', '/home/northnpk/Downloads/JetClass_dataset/train/HToCC_035.root']
    train_path.sort()
    # start = timeit.default_timer()
    all_chunk_x_part = []
    all_chunk_x_jet = []
    all_chunk_y = []
    num_files = 20
    print(f'num_files: {num_files}')
    for file in train_path[:num_files]:
        (x_part, x_jet, y) = chunk_processing(file, max_num_particles, particle_features, jet_features, labels)
        all_chunk_x_part.extend(x_part)
        all_chunk_x_jet.extend(x_jet)
        all_chunk_y.extend(y)
    # # Multi-Processor to get all chunk of dataset
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(chunk_processing, file, max_num_particles, particle_features, jet_features, labels ) for file in train_path[:10]]
    #     for future in futures:
    #         (x_part, x_jet, y) = future.result()
    #         all_chunk_x_part.extend(x_part)
    #         all_chunk_x_jet.extend(x_jet)
    #         all_chunk_y.extend(y)
    dataloader = DataLoader(dataset=TensorDataset(
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_part))), 
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_x_jet))), 
        torch.from_numpy(ak.to_numpy(np.concatenate(all_chunk_y)))), 
        batch_size=batch_size, shuffle=True)
    # print(f'Awkward+Uproot init time count: {timeit.default_timer()-start}')
    return dataloader
    
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

@profile
# Define Training loop
def train(model, loss_fn, optimizer, train_loader):
    sum_loss = 0
    for (x_part, x_jet, y) in train_loader:
        # print(x_part.shape)
        # pass
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
    model = SimpleJetClassModel()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # print(model)
    # summary(model, [(1, 4, 128,), (1, 4,)], device='cpu')

    # Create Dataloader
    print("Preparing Dataloader from Uproot")
    train_loader = load_dataset()
    # print("Done")

    # Train 1 Epoch
    epoch = 1
    print(f"Start training: {epoch} epoch")
    # start = timeit.default_timer()
    # train(model, loss_fn, optimizer, train_loader)
    for _ in range(epoch): train(model, loss_fn, optimizer, train_loader)
    # print(f'Training time count: {timeit.default_timer()-start}')
    # print("Done")
