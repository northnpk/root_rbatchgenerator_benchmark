import ROOT
 
filename = ROOT.gROOT.GetTutorialDir().Data() + "/dataframe/df017_vecOpsHEP.root"
treename = "myDataset"

def WithRDataFrameVecOpsJit(treename, filename):
    f = ROOT.RDataFrame(treename, filename)
    h = f.Define("good_pt", "sqrt(px*px + py*py)[E>100]")\
         .Histo1D(("pt", "With RDataFrame and RVec", 16, 0, 4), "good_pt")
    h.DrawCopy()
 
## We plot twice the same quantity, the key is to look into the implementation
## of the functions above.
c = ROOT.TCanvas()
c.Divide(2,1)
c.cd(2)
WithRDataFrameVecOpsJit(treename, filename)
# c.SaveAs("df017_vecOpsHEP.png")
 
print("Saved figure to df017_vecOpsHEP.png")