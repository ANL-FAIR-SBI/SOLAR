import h5py
import pickle
import os

folder_dir = '/path/to/data/cosmoUniverse_21688988'
saving_path = '/path/to/idx/list/'
with open(saving_path+"file_lists", "rb") as fp:
    file_list=pickle.load(fp)
h5_filename = os.path.join(
    folder_dir,
    file_list[100]
)
with h5py.File(h5_filename, 'r') as f:
    print(f.keys())
    x=f['full'][:]
    y=f['unitPar'][:]
    print(x.shape)
    print(y.shape)
print(len(file_list))

