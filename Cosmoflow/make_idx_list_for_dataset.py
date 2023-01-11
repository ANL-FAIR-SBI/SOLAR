import os
import pickle
#Define the folder location
folder_dir = '/path/to/data/cosmoUniverse_21688988'
saving_path = '/path/to/idx/list/'
#List all the filenames under this directory and save to numpy.
#result = os.listdir(folder_dir)
result = sorted( filter( lambda x: os.path.isfile(os.path.join(folder_dir, x)),
                        os.listdir(folder_dir) ) )

with open(saving_path+"file_lists", "wb") as fp:
        pickle.dump(result,fp)