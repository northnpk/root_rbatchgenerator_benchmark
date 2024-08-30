import torch
import ROOT
from tqdm.auto import tqdm
import os

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

# Returns two generators that return training and validation batches
# as PyTorch tensors.
print('Getting Generator for train set of JetClass datasets ...')
gen_train, _ = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    tree_name,
    train_path,
    batch_size,
    chunk_size,
    columns=columns+vec_columns+targets,
    max_vec_sizes=max_vec_sizes,
    targets=targets,
    validation_split=0,
)

print('Getting Generator for validation set of JetClass datasets ...')
gen_validate, _ = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    tree_name,
    val_path,
    batch_size,
    chunk_size,
    columns=columns+vec_columns+targets,
    max_vec_sizes=max_vec_sizes,
    targets=targets,
    validation_split=0,
)

print('Getting Generator for test set of JetClass datasets ...')
gen_test, _ = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    tree_name,
    test_path,
    batch_size,
    chunk_size,
    columns=columns+vec_columns+targets,
    max_vec_sizes=max_vec_sizes,
    targets=targets,
    validation_split=0,
)

# Get a list of the columns used for training
input_columns = gen_train.train_columns
num_features = len(input_columns)
num_targets = len(gen_train.target_columns)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(device=device)
# print(f'Using device: {device}')

print('Start training...')
 # Loop through the training set and train model
for i, (x_train, y_train) in tqdm(enumerate(gen_train), total=780000):
    # Make prediction and calculate loss
    # pred = model(x_train.to(device))
    # print(pred.to_numpy)
    # loss = loss_fn(pred, y_train.to(device))
    pred = model(x_train)
    loss = loss_fn(pred, y_train)
 
    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()
    # break
 
    # Calculate accuracy
    # accuracy = calc_accuracy(y_train, pred)
 
    # print(f"Training => loss: {loss}")
print('Finished training')
# print('Start Testing...')

# #################################################################
# # Validation
# #################################################################
 
# # Evaluate the model on the validation set
# for i, data in tqdm(enumerate(gen_test), total=156000):
#     # Make prediction and calculate accuracy
#     x_train, y_train = data.to(device)
#     pred = model(x_train)
#     # loss = loss_fn(pred, y_train)
#     # accuracy = calc_accuracy(y_train, pred)
 
#     # print(f"Validation => loss: {loss}")
#     # break

print('Ended')
