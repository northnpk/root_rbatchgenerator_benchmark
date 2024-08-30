import torch
import ROOT
from ROOT import RDataFrame
from tqdm.auto import tqdm
import os
import timeit

tree_name = "tree"
# file_name = "/home/northnpk/Downloads/test_20M/HToCC_100.root"
# tree_name = "sig_tree"
# file_name = "http://root.cern/files/Higgs_data.root"
batch_size = 128
chunk_size = 5_000
vec_columns=['part_px', 'part_py', 'part_pz', 'part_energy']
columns=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
# columns.extend(vec_columns)
max_vec_sizes = dict(zip(vec_columns, [128]*4))
# target = 'Type'
# targets = 'label_Hcc'
# targets = ['label_QCD', 'label_Hbb', 'label_Hcc']
targets = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']

print('Getting path of JetClass datasets ...')
data_path = '/home/northnpk/Downloads/JetClass_dataset/'
train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]
val_path = [data_path + 'val/' + p for p in os.listdir(data_path + 'val')]
test_path = [data_path + 'test/' + p for p in os.listdir(data_path + 'test')]
# print(f'Train path: {train_path}')
# print(f'Validation path: {val_path}')
# print(f'Test path: {test_path}')

# Getting RDataFrame from multiple root filepath
names = ROOT.std.vector('string')()


data_path = '/home/northnpk/Downloads/JetClass_dataset/'
train_path = [data_path + 'train/' + p for p in os.listdir(data_path + 'train')]
for n in train_path[:10]: names.push_back(n)

r_df = RDataFrame(tree_name, names)

# Returns two generators that return training and validation batches
# as PyTorch tensors.
print('Getting Generator for train set of JetClass datasets ...')
start = timeit.default_timer()
gen_train = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    r_df,
    batch_size,
    chunk_size,
    columns=columns+vec_columns+targets,
    max_vec_sizes=max_vec_sizes,
    target=targets,
    validation_split=0,
)
print(f'RBatchGenerator time count {timeit.default_timer()-start}')

# Get a list of the columns used for training
input_columns = gen_train.train_columns
num_features = len(input_columns)
num_targets = len(gen_train.target_columns)

print(f'Number of input columns: {num_features}')
print(f'Number of targets columns: {num_targets}')

from torch.utils.data import DataLoader, IterableDataset
class RBatchDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator
        self.batch_size = self.generator.base_generator.batch_size
        # print(f'Batch size: {self.batch_size}')
        if self.generator.last_batch_no_of_rows != 0 and self.generator.number_of_batches > 1:
            self.length = ((self.generator.number_of_batches-1) * self.batch_size) + self.generator.last_batch_no_of_rows
        elif self.generator.number_of_batches == 1:
            self.length = self.generator.last_batch_no_of_rows
        else :
            self.length = (gen_train.number_of_batches * self.batch_size)

    def __iter__(self):
        return self.generator.__iter__()
    
    def __len__(self):
        return self.generator.number_of_batches

loader = DataLoader(dataset=RBatchDataset(gen_train), batch_size=None)

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
        # input 4x43x3, output 32x43x3
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x43x3, output 32x43x3
        x = self.act2(self.conv2(x))
        # input 32x43x3, output 32x21x1
        x = self.pool2(x)
        # input 32x21x1, output 672
        x = self.flat(x)
        # input 672, output 512
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
    # for i, (x_train, y_train) in enumerate(tqdm(gen_train)):
    for i, (x_train, y_train) in enumerate(tqdm(loader)):
        # Make prediction and calculate loss
        tmp = (x_train, y_train)
        # print(tmp)
    #     pred = model(x_train)
    #     # print(pred.to_numpy)
    #     loss = loss_fn(pred, y_train.to(torch.float32))
    
    #     # improve model
    #     model.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    
    #     # Calculate accuracy
    #     sum_loss += loss.item()*y_train.size(0)

    # print(f"Training => loss: {sum_loss/loader.dataset.length}")
# import memray

# for e in range(10):
#     if e in [0,1,9]:
#         with memray.Tracker(f"rbatchgentorch_epoch_{e}.bin", native_traces=True):
#             train()
#     else:
#         train()

print(f"tmvaRBatchGenerator Training time avg.: {timeit.timeit('train()', number=3, globals=globals())/10.0}")
print('Finished training')
print('Ended')
