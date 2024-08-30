import ROOT
 
tree_name = "sig_tree"
file_name = "http://root.cern/files/Higgs_data.root"
df = ROOT.RDataFrame(tree_name, [file_name]*100)
batch_size = 128
chunk_size = 5_000
 
ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    df,
    batch_size,
    chunk_size,
    validation_split=0.3,
    shuffle=True,
)
 
# Loop through training set
for i, b in enumerate(ds_train):
    print(f"Training batch {i} => {b.shape}")