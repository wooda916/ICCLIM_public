
# coding: utf-8

# In[1]:

import xarray as xr
import dask.array as dsa
import zarr
import glob
from rechunker import rechunk
import numpy as np


# In[2]:

import os
import sys


# In[3]:

root     = '/fastdata/ci1twx/'
variable = 'tasmax'
exp      = 'SSP585'


# In[4]:

directory      = f'{root}{variable}/{exp}/'
directory_list = glob.glob(directory + '*')


# In[7]:

model_list = [d.split(f'{exp}/')[1] for d in directory_list]


# In[11]:

def get_r_folders(model):
    if 'SSP' in exp:
        r_dirs = glob.glob(f'{directory}{model}/*')
        r_list = [d.split(f'{model}/')[1] for d in r_dirs]
    else:
        r_list = ['r1i1p1f1']
    return sorted(r_list)


# In[13]:

def find_array_size(model):
    r_list = get_r_folders(model)
    coords = {}
    for r in r_list:
        coords[r] = {}
        files = glob.glob(f'{directory}{model}/{r}/*.nc')
        
        # determine if all files have same grid:
        grid = files[0][-24:-21]
        grid_check = [True if grid in f else False for f in files]
        
        if all(grid_check) == True: # if all grids are the same
            coords[r][str(grid)] = {}
            try:
                ds = xr.open_mfdataset(files, use_cftime=True)
                lats = ds.coords['lat'].values.shape
                lons = ds.coords['lon'].values.shape
                time = ds.coords['time'].values.shape
                coords[r][grid]['lats'] = lats
                coords[r][grid]['lons'] = lons
                coords[r][grid]['time'] = time
            except:
                print(f'{model} failed...')
            
        else: # if there is >1 grid
            #check how many grids there are and add them to a list
            grid_list = [grid]
            for f in files:
                grid = f[-24:-21]
                if grid not in grid_list:
                    grid_list.append(grid) 
            for g in grid_list:
                coords[r][str(g)] = {}
                grid_files = []
                for f1 in files:
                    if g in f1:
                        grid_files.append(f1)
                try:
                    ds = xr.open_mfdataset(grid_files, use_cftime=True)
                    lats = ds.coords['lat'].values.shape
                    lons = ds.coords['lon'].values.shape
                    time = ds.coords['time'].values.shape
                    coords[r][g]['lats'] = lats
                    coords[r][g]['lons'] = lons
                    coords[r][g]['time'] = time
                except:
                    print(f'{model} failed...')
    return coords


# In[ ]:

# Create dictionary of model grid sizes for all models
m_grids = {}
for i, m in enumerate(model_list):
    print (f'Model {i} of {len(model_list)}:  {m}')
    m_grids[m] = find_array_size(m)


# In[ ]:

# Save dictionary of model grid sizes to a numpy file 
#np.save('/fastdata/ci1twx/numpy_output/tasmax_ssp585_model_grid_sizes.npy', m_grids)


# In[15]:

# Load dictionary of model grid sizes to a numpy file
#m_grids = np.load('/fastdata/ci1twx/numpy_output/tasmax_ssp585_model_grid_sizes.npy', allow_pickle=True).item()


# In[17]:

# get a common factor for grid coordinates to create equal rechunks
from math import gcd
def cf(num1,num2):
    n=[]
    g=gcd(num1, num2)
    for i in range(1, g+1): 
        if g%i==0: 
            n.append(i)
    return n


# In[18]:

# Create dictionary of grid sizes and factor by which to rechunk each of the models data dimensions e.g. 100/4 = 25 (blocks of 25)
gridz = {}
rechunk_factor = {}

for m in model_list:
    gridz[m] = {}
    rechunk_factor[m] = {}
    for r in m_grids[m].keys():
        gridz[m][r] = {}
        rechunk_factor[m][r] = {}
        for g in m_grids[m][r].keys():
            gridz[m][r][g] = []
            #rechunk_factor[m][r][g] = []
            if not [m_grids[m][r][g]['lats'], m_grids[m][r][g]['lons']] in gridz[m][r][g]:
                gridz[m][r][g].append([m_grids[m][r][g]['lats'], m_grids[m][r][g]['lons']])
                myNumber = 6
                if m != 'IPSL-CM6A-LR':
                    rechunk_factor[m][r][g] = min(cf(m_grids[m][r][g]['lats'][0],m_grids[m][r][g]['lons'][0]), key=lambda x:abs(x-myNumber))
                else:
                    rechunk_factor[m][r][g] = 6


# In[19]:

# Define output folder for output zarr files
zarr_output = f'/fastdata/ci1twx/rechunked/{exp}/{variable}/zarr/'


# In[20]:

# Configure Dask
import dask
dask.config.set({"array.slicing.split_large_chunks": False})


# In[21]:

def rechunk_model(model):
    r_list = get_r_folders(model)
    for r in r_list:
        files = glob.glob(f'{directory}{model}/{r}/*.nc')
        # determine if all files have same grid:
        grid = files[0][-24:-21]
        grid_check = [True if grid in f else False for f in files]
        
        if all (grid_check) == True:
            try:
                ds = xr.open_mfdataset(files, use_cftime=True)
                lats = ds.coords['lat'].values.shape[0]
                lons = ds.coords['lon'].values.shape[0]
                # convert calendar to ignore leap years
                ds2 = ds.convert_calendar("360_day", align_on = "year")
                # rechunk to 20 years and chunks determined by grid size (as above)
                n = 7200 # 360 * 20 years
                n_lat = int(lats/rechunk_factor[model][r][grid])
                n_lon = int(lons/rechunk_factor[model][r][grid])
                ds3 = ds2.chunk({'time': n, 'lat': n_lat, 'lon': n_lon})
                print(ds3.tasmax.data)
                
                ds3.tasmax.encoding = {} # helps when writing to zarr
                print("encoding complete")
                
                # Define model output folder
                folder = zarr_output + model
                # Check if output folder exists for model, if not create it
                if not os.path.exists(folder):
                    os.makedirs(folder)
                print ("output folder created")
                
                # Define r output folder
                r_folder = f'{folder}/{r}'
                # Check if folder for r exists, if not create it
                if not os.path.exists(r_folder):
                    os.makedirs(r_folder)
                print("r folder created")
                
                print('***    ALL folders created...')                
                
                #! rm -rf /fastdata/ci1twx/rechunked/SSP585/tasmax/zarr/*/*.zarr # clean up any existing temporary data
                
                ds3.to_zarr(f'{r_folder}/{model}_{exp}_{variable}_rechunked_20yrs.zarr')
                print('zarr created ...')
                source_group = zarr.open(f'{r_folder}/{model}_{exp}_{variable}_rechunked_20yrs.zarr')
                print('zarr opened ...')
                source_array = source_group[variable]
                target_chunks = {
                    'tasmax': {'time': n, 'lat': n_lat, 'lon': n_lon},
                    'time': None, # don't rechunk this array
                    'lon': None,
                    'lat': None,
                }

                max_mem = '100MB'

                target_store = f'{r_folder}/{model}_{exp}_{variable}_rechunked_20yrs_out.zarr'
                temp_store = f'{r_folder}/{model}_{exp}_{variable}_rechunked_20yrs_out-temp.zarr'
                
                # "need to have a variable bash command for the next line
                #!rm -rf /fastdata/ci1twx/rechunked/SSP585/tasmax/zarr/*/*.zarr
                array_plan = rechunk(source_group, target_chunks, max_mem, target_store, temp_store=temp_store)
                print('array plan created ...')
                #array_plan
                array_plan.execute()#
                print('array plan executed ...')
            except:
                print(f'{model} failed...')
        else:
            grid_list = [grid]
            for f in files:
                grid = f[-24:-21]
                if grid not in grid_list:
                    grid_list.append(grid)
                for g in grid_list:
                    grid_files = []
                    for f1 in files:
                        if g in f1:
                            grid_files.append(f1)
                    try:
                        ds = xr.open_mfdataset(grid_files, use_cftime=True)
                        lats = ds.coords['lat'].values.shape[0]
                        lons = ds.coords['lon'].values.shape[0]
                        # convert calendar to ignore leap years
                        ds2 = ds.convert_calendar("360_day", align_on = "year")
                        # rechunk to 20 years and chunks determined by grid size (as above)
                        n = 7200 # 360 * 20 years
                        n_lat = int(lats/rechunk_factor[model][r][g])
                        n_lon = int(lons/rechunk_factor[model][r][g])
                        ds3 = ds2.chunk({'time': n, 'lat': n_lat, 'lon': n_lon})
                        print(ds3.tasmax.data)
                        ds3.tasmax.encoding = {} # helps when writing to zarr
                        print("encoding complete")
                        
                        # Define model output folder
                        folder = zarr_output + model
                        # Check if output folder exists for model, if not create it
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        print ("output folder created")
                            
                        # Define r output folder
                        r_folder = f'{folder}/{r}'
                        # Check if folder for r exists, if not create it
                        if not os.path.exists(r_folder):
                            os.makedirs(r_folder)
                        print("r folder created")
                            
                        # define grid output folder
                        grid_folder = f'{r_folder}/{g}'
                        # Check if folder for grid exists, if not create it
                        if not os.path.exists(grid_folder):
                            os.makedirs(grid_folder)
                        print ('grid folder created')
                        
                        print('***   ALL folders created...')
                        
                        #! rm -rf /fastdata/ci1twx/rechunked/SSP585/tasmax/zarr/*/*.zarr # clean up any existing temporary data
                        
                        ds3.to_zarr(f'{grid_folder}/{model}_{exp}_{variable}_rechunked_20yrs.zarr')
                        print('zarr created ...')
                        source_group = zarr.open(f'{grid_folder}/{model}_{exp}_{variable}_rechunked_20yrs.zarr')
                        print('zarr opened ...')
                        source_array = source_group[variable]
                        
                        target_chunks = {
                            'tasmax': {'time': n, 'lat': n_lat, 'lon': n_lon},
                            'time': None, # don't rechunk this array
                            'lon': None,
                            'lat': None,
                        }

                        max_mem = '100MB'

                        target_store = f'{grid_folder}/{model}_{exp}_{variable}_rechunked_20yrs_out.zarr'
                        temp_store = f'{grid_folder}/{model}_{exp}_{variable}_rechunked_20yrs_out-temp.zarr'

                        # "need to have a variable bash command for the next line"
                        #!rm -rf /fastdata/ci1twx/rechunked/SSP585/tasmax/zarr/*/*.zarr
                        array_plan = rechunk(source_group, target_chunks, max_mem, target_store, temp_store=temp_store)
                        print('array plan created ...')

                        #array_plan
                        array_plan.execute()
                        print('array plan executed ...')

                    except:
                        print(f'{model} failed...')                        


# In[23]:

#model_list.remove('GFDL-CM4')
#model_list.remove('AWI-CM-1-1-MR')
#model_list.remove('CESM2')
#model_list.remove('NorESM2-MM')


# In[ ]:

# Execute rechunking
for i, m in enumerate(model_list):
    print (f'Model {i} of {len(model_list)}:  {m}')
    rechunk_model(m)
    print (f'{m} finished...\n')
    
print ("*** ALL MODELS RECHUNKED ***")

