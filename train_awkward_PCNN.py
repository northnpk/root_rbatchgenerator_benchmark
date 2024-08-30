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
class ResNetUnit(torch.nn.Module):
    r"""Parameters
    ----------
    in_channels : int
        Number of channels in the input vectors.
    out_channels : int
        Number of channels in the output vectors.
    strides: tuple
        Strides of the two convolutional layers, in the form of (stride0, stride1)
    """

    def __init__(self, in_channels, out_channels, strides=(1, 1), **kwargs):

        super(ResNetUnit, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.dim_match = True
        if not in_channels == out_channels or not strides == (1, 1):  # dimensions not match
            self.dim_match = False
            self.conv_sc = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                     stride=strides[0] * strides[1], bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print('resnet unit', identity.shape, x.shape, self.dim_match)
        if self.dim_match:
            return identity + x
        else:
            return self.conv_sc(identity) + x


class ResNet(torch.nn.Module):
    r"""Parameters
    ----------
    features_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    conv_params : list
        List of the convolution layer parameters. 
        The first element is a tuple of size 1, defining the transformed feature size for the initial feature convolution layer.
        The following are tuples of feature size for multiple stages of the ResNet units. Each number defines an individual ResNet unit.
    fc_params: list
        List of fully connected layer parameters after all EdgeConv blocks, each element in the format of
        (n_feat, drop_rate)
    """

    def __init__(self, features_dims, num_classes,
                 conv_params=[(32,), (64, 64), (64, 64), (128, 128)],
                 fc_params=[(512, 0.2)],
                 for_inference=False,
                 **kwargs):

        super(ResNet, self).__init__(**kwargs)
        self.conv_params = conv_params
        self.num_stages = len(conv_params) - 1
        self.fts_conv = torch.nn.Sequential(
            torch.nn.BatchNorm1d(features_dims),
            torch.nn.Conv1d(
                in_channels=features_dims, out_channels=conv_params[0][0],
                kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(conv_params[0][0]),
            torch.nn.ReLU())

        # define ResNet units for each stage. Each unit is composed of a sequence of ResNetUnit block
        self.resnet_units = torch.nn.ModuleDict()
        for i in range(self.num_stages):
            # stack units[i] layers in this stage
            unit_layers = []
            for j in range(len(conv_params[i + 1])):
                in_channels, out_channels = (conv_params[i][-1], conv_params[i + 1][0]) if j == 0 \
                    else (conv_params[i + 1][j - 1], conv_params[i + 1][j])
                strides = (2, 1) if (j == 0 and i > 0) else (1, 1)
                unit_layers.append(ResNetUnit(in_channels, out_channels, strides))

            self.resnet_units.add_module('resnet_unit_%d' % i, torch.nn.Sequential(*unit_layers))

        # define fully connected layers
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            in_chn = conv_params[-1][-1] if idx == 0 else fc_params[idx - 1][0]
            fcs.append(torch.nn.Sequential(torch.nn.Linear(in_chn, channels), torch.nn.ReLU(), torch.nn.Dropout(drop_rate)))
        fcs.append(torch.nn.Linear(fc_params[-1][0], num_classes))
        if for_inference:
            fcs.append(torch.nn.Softmax(dim=1))
        self.fc = torch.nn.Sequential(*fcs)

    def forward(self, features, jet_features):
        # x: the feature vector, (N, C, P)
        x = self.fts_conv(features)
        for i in range(self.num_stages):
            x = self.resnet_units['resnet_unit_%d' % i](x)  # (N, C', P'), P'<P due to kernal_size>1 or stride>1

        # global average pooling
        x = x.mean(dim=-1)  # (N, C')
        # fully connected
        x = self.fc(x)  # (N, out_chn)
        return x


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
    conv_params = [(32,), (64, 64), (64, 64), (128, 128)]
    fc_params = [(512, 0.2)]

    pf_features_dims = 4
    num_classes = 10
    model = ResNet(pf_features_dims, num_classes,
                   conv_params=conv_params,
                   fc_params=fc_params)
    
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