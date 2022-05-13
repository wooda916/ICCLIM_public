
# coding: utf-8

# In[1]:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:

## Load modules
import pyproj as pyp

from libpysal.weights import lat2W

import icclim

import sys
import glob
import os
import datetime
import cftime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import nc_time_axis
import math

import cf, cfplot as cfp

#print("python: ",sys.version)
#print("numpy: ", np.__version__)
#print("xarray: ", xr.__version__)
#print("pandas: ", pd.__version__)
#print("icclim: ", icclim.__version__)


# In[22]:

# Setting Dask options
#"https://icclim.readthedocs.io/en/latest/how_to/dask.html"

import dask
from distributed import Client
memory_limit       = '16GB'
n_workers          = 1
threads_per_worker = 2

# --OPTION 1--
client = Client(memory_limit=memory_limit, n_workers=n_workers, threads_per_worker=threads_per_worker)
dask.config.set({"array.slicing.split_large_chunks": False})
dask.config.set({"array.chunk-size": "100 MB"})
dask.config.set({"distributed.worker.memory.target": "0.8"})
dask.config.set({"distributed.worker.memory.spill": "0.9"})
dask.config.set({"distributed.worker.memory.pause": "0.95"})
dask.config.set({"distributed.worker.memory.terminate": "0.98"})


# In[20]:

# Define input and output file locations
root        = '/shared/rise_group/User/ci1twx/CMIP6/'
output_path = '{0}OUTPUT/{1}/'.format(root,index)


# In[21]:

# Define index, variable and experiment
variable = 'tasmax'
exp      = 'SSP585'
index    = 'TX90p'


# In[22]:

# Make list of available directories for the selected variable and experiment
directory      = '{0}{1}/{2}/'.format(root, exp, variable)
directory_list = glob.glob(directory + '*')


# In[23]:

# Make list of models available
model_list = [d.split('{0}/'.format(variable))[1] for d in directory_list]


# In[ ]:

def compute_index(index, variable, exp, m, date_range):
    # Create list of experiment simulation variants (e.g. r1i1p1f1) available
    if 'SSP' in exp:
        r_dirs = glob.glob('{0}{1}/*'.format(directory,m))
        r_list = [d.split('{0}/'.format(m))[1] for d in r_dirs]
    else:
        r_list = ['r1i1p1f1']
        
    # Define model output folder
    folder = output_path + m
    # Check if output folder exists for model, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Iterate over simulation variants (r numbers)
    for i, r in enumerate(r_list):
        # Define r output folder
        r_folder = '{0}/{1}'.format(folder, r)
        # Check if folder for r exists, if not create it
        if not os.path.exists(r_folder):
            os.makedirs(r_folder)
                
        # Check if any files that contain the defined experiment name already exist in the r folder
        # If no files exist, proceed with processing
        if not glob.glob('{0}/*{1}*'.format(r_folder, exp)):
            print('Processing...\nModel: {0}\nR: {1}'.format(m,r))
            if 'SSP' in exp:
                input_filenames = glob.glob('{0}{1}/{2}/*.nc'.format(directory, m, r)
                output_filename = '{0}/{1}_{2}_{3}_{4}_{5}.nc'.format(r_folder, index, m, exp, r, date_range)
            else:
                input_filenames = glob.glob('{0}{1}/*.nc'(directory, m))[:20] # just select the first 20 files to speed things up for now
                output_filename = '{0}/{1}_{2}_{3}_{4}_{5}.nc'.format(r_folder, index, m, exp, r, 'full_range')
                
            try:
                icclim.index(index_name=index, in_files=input_filenames, var_name=variable, slice_mode='year', out_file=output_filename, logs_verbosity='LOW')
                print ('{0} COMPLETED!\n'.format(m))
            except:
                print ('{0} FAILED!\n'.format(m))


# In[ ]:

# Iterate over models
for m in model_list:
    compute_index(index=index, variable=variable, exp=exp, m=m, date_range = '2015-2100')


# In[34]:

# Save script to .py
get_ipython().system('jupyter nbconvert --to script ICCLIM_indices-TX90p.ipynb')

