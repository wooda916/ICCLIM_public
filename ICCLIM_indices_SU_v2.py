from datetime import datetime
# load datetime object so that timestamps can be printed
dateTimeObj = datetime.now()
print(dateTimeObj)

print ("loading modules...")
## Load modules
import pyproj as pyp

from libpysal.weights import lat2W

import icclim

import sys
import glob
import os
import datetime
import cftime
from os.path import exists

import numpy as np
import pandas as pd
import xarray as xr
import nc_time_axis
import math

import argparse
import traceback

print("modules loaded ...")
print(dateTimeObj)

print("parse arguments")
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--RAM", required=True)
parser.add_argument("-c", "--cores", required=True)
parser.add_argument("-v", "--variable", required=True)
parser.add_argument("-e", "--experiment", required=True)
parser.add_argument("-i", "--index", required=True)

args = parser.parse_args()
print("arguments parsed...")
print(dateTimeObj)


# Setting Dask options
#"https://icclim.readthedocs.io/en/latest/how_to/dask.html"
print("Configuring Dask ...")
print(dateTimeObj)
import dask
from distributed import Client
memory_limit       = f'{args.RAM}GB'
n_workers          = 1
threads_per_worker = args.cores # n cores, SMP, 16GB per core
print(memory_limit)
print(n_workers)


# --OPTION 1--

if __name__ == '__main__':
    print ("setting client...")
    client = Client(memory_limit=memory_limit, n_workers=n_workers, threads_per_worker=threads_per_worker)
    print("client set!")
    #client = Client(memory_limit=memory_limit, n_workers=n_workers, threads_per_worker=threads_per_worker)
    dask.config.set({"array.slicing.split_large_chunks": False})
    dask.config.set({"array.chunk-size": "100 MB"})
    dask.config.set({"distributed.worker.memory.target": "0.8"})
    dask.config.set({"distributed.worker.memory.spill": "0.9"})
    dask.config.set({"distributed.worker.memory.pause": "0.95"})
    dask.config.set({"distributed.worker.memory.terminate": "0.98"})
    print("Dask configured!")
    print(dateTimeObj)

    # Define index, variable and experiment
    variable = str(args.variable) #'tasmax'
    exp      = str(args.experiment) #'SSP585'
    index    = str(args.index) #'TX90p'

    # Define index, variable and experiment
    #variable = 'tasmax'
    #exp      = 'SSP585'
    #index    = 'TX90p'

    print("ARGUMENTS:")
    print("mem limit = ", memory_limit)
    print("n_workers = ", n_workers)
    print("threads =  ", threads_per_worker)

    print ("variable = ", variable)
    print ("exp = ", exp)
    print ("index = ", index)

    # inout paths
    in_root = f'/fastdata/ci1twx/{variable}/{exp}/'

    # output paths
    out_root_zarr = f'/fastdata/ci1twx/OUTPUT/{exp}/{variable}/zarr/'
    out_root_nc = f'/fastdata/ci1twx/OUTPUT/{exp}/{variable}/NetCDF/'

    # Make list of available directories for the selected variable and experiment
    directory_list = glob.glob(in_root + '*')

    # Make list of models available
    model_list = [d.split(f'{exp}/')[1] for d in directory_list]

    ref   = [2015, 2025]
    study = [2075, 2095]

    ref_period   = [datetime.datetime(ref[0], 1, 1), datetime.datetime(ref[1], 12, 30)]
    study_period = [datetime.datetime(study[0], 1, 1), datetime.datetime(study[1], 12, 31)]

    def folder_check(list_folders):
        for i in list_folders:
            if not os.path.exists(i):
                os.makedirs(i)

    def grid_check(filepath):
        files = glob.glob(filepath)
        grid = files[0][-24:-21]
        check = [True if grid in f else False for f in files]
        check_bool = True if all (check) == True else False
        return check_bool

    def create_grid_list(in_files):
        grid_list = []
        for f in in_files:
            grid = f[-24:-21]
            if grid not in grid_list:
                grid_list.append(grid)
        return grid_list

    # compute percentile indices
    def compute_index_p(index, variable, exp, m, ref_period, study_period):
        # Create list of experiment simulation variants (e.g. r1i1p1f1) available

        if 'SSP' in exp:
            r_dirs = glob.glob(f'{in_root}{m}/*')
            r_list = [d.split(f'{m}/')[1] for d in r_dirs]
        else:
            r_list = ['r1i1p1f1']
        for r in r_list:
            print (f"processing {r} ...")
            # list input files
            in_filepath = f'{in_root}{m}/{r}/*'
            in_files    = glob.glob(in_filepath)

            # Define model output folder
            zarr_folder = f'{out_root_zarr}{m}/{r}'
            nc_folder   = f'{out_root_nc}{m}/{r}'

            # Check if output folder exists for model, if not create it
            folder_check([zarr_folder, nc_folder])
            if grid_check(in_filepath) == True: # if all grids are the same
                # check if zarr files already exist
                zarr_file_exists = exists(f"{zarr_folder}/{args.variable}_opti.zarr")
                if zarr_file_exists == True:
                    try:
                        print ("zarr exists - begin processing...")
                        print(dateTimeObj)
                        icclim.index(
                                index_name             = index,
                                in_files               = f"{zarr_folder}/{args.variable}_opti.zarr",
                                slice_mode             = "year",
                                ignore_Feb29th         = True,
                                base_period_time_range = ref_period,
                                time_range             = study_period,
                                out_file               = f"{nc_folder}/{args.index}_20yr.nc",
                            )
                        print ("COMPLETE!!!")
                        print(dateTimeObj)
                    except Exception as e:
                        print(f'{m} failed...')
                        print(dateTimeObj)
                        print(e)
                else:
                    try:
                        print ("begin processing...")
                        print(dateTimeObj)
                        with icclim.create_optimized_zarr_store(
                            in_files               = in_files,
                            var_names              = variable,
                            target_zarr_store_name = f"{zarr_folder}/{args.variable}_opti.zarr",
                            keep_target_store      = True,
                            chunking               = {"time": -1, "lat": "auto", "lon": "auto"},
                        ) as opti_tas:
                             icclim.index(
                                index_name             = index,
                                in_files               = opti_tas,
                                slice_mode             = "year",
                                ignore_Feb29th         = True,
                                base_period_time_range = ref_period,
                                time_range             = study_period,
                                out_file               = f"{nc_folder}/{args.index}_20yr.nc",
                            )
                        print ("COMPLETE!!!")
                        print(dateTimeObj)

                    except Exception as e:
                        print(f'{m} failed...')
                        print(dateTimeObj)
                        print(e)
            else:
                grid_list = create_grid_list(in_files)
                for g in grid_list:
                    print (f"processing grid: {g} ...")
                    grid_files = []
                    for f1 in in_files:
                        if g in f1:
                            grid_files.append(f1)
                    zarr_folder_grid = f'{zarr_folder}/{g}'
                    nc_folder_grid   = f'{nc_folder}/{g}'
                    folder_check([zarr_folder_grid, nc_folder_grid])
                    # Check if zarr file already exists
                    zarr_file_exists = exists(f"{zarr_folder}/{args.variable}_opti.zarr")
                    if zarr_file_exists == True:
                        try:
                            print ("zarr exists - begin processing...")
                            print(dateTimeObj)
                            icclim.index(
                                index_name             = index,
                                in_files               = f"{zarr_folder}/{args.variable}_opti.zarr",
                                slice_mode             = "year",
                                ignore_Feb29th         = True,
                                base_period_time_range = ref_period,
                                time_range             = study_period,
                                out_file               = f"{nc_folder_grid}/{args.index}_20yr.nc",
                            )
                        except Exception as e:
                            print(f'{m} failed...')
                            print(dateTimeObj)
                            print(e)
                    else:
                        try:
                            print ("begin processing...")
                            print(dateTimeObj)
                            with icclim.create_optimized_zarr_store(
                                in_files               = grid_files,
                                var_names              = variable,
                                target_zarr_store_name = f"{zarr_folder_grid}/{args.variable}_opti.zarr",
                                keep_target_store      = True,
                                chunking               = {"time": -1, "lat": "auto", "lon": "auto"},
                            ) as opti_tas:
                                 icclim.index(
                                    index_name             = index,
                                    in_files               = opti_tas,
                                    slice_mode             = "year",
                                    ignore_Feb29th         = True,
                                    base_period_time_range = ref_period,
                                    time_range             = study_period,
                                    out_file               = f"{nc_folder_grid}/{args.index}_20yr.nc",
                                )
                            print ("COMPLETE!!!")
                            print(dateTimeObj)
                        except Exception as e:
                            print(f'{m} failed...')
                            print(dateTimeObj)
                            print(e)

    # execute but check if models have already been processed first
    # by checking if the NetCDF file already exists
    from os.path import exists

    # Catch errors and write to a txt file:
    with open(f"/fastdata/ci1twx/batch_jobs/errors/{args.index}_log.txt", "w") as log:
        try:
            for i, m in enumerate(model_list):
                print (f'Model {i} of {len(model_list)}:  {m}')
                folder_exists = f'{out_root_nc}{m}'
                if os.path.exists(folder_exists):
                    print ("Model NetCDF folder exits.  Checking r numbers...")
                    r_list = glob.glob(f'{folder_exists}/*')
                    print(r_list)
                    for r in r_list:
                        print(r)
                        # check if there are g numbers
                        r_files = glob.glob(f'{r}/*')
                        print('r_files', r_files)
                        if r_files and "g" in r_files[0][-3:]:
                            print("grid exists...")
                            for g in r_files:
                                print(g)
                                g_files = glob.glob(f'{g}/*')
                                print("filepath = ", f'{g}/{args.index}_20yr.nc')
                                file_exists = exists(f'{g}/{args.index}_20yr.nc')
                                print('file:  ', file_exists)
                                if file_exists == True:
                                    print ("file exists")
                                    #break
                                else:
                                    print ("COMPUTING")
                                    compute_index_p(index=index, variable=variable, exp=exp, m=m, ref_period=ref_period, study_period=study_period)
                                    print (f'{m} finished...\n')
                        else:
                            print ("grid does not exist")
                            print("filepath = ", f'{r}/{args.index}_20yr.nc')
                            file_exists = exists(f'{r}/{args.index}_20yr.nc')
                            print('file:  ', file_exists)
                            if file_exists == True:
                                print ("file exits")
                                #break
                            else:
                                print ("COMPUTING")
                                compute_index_p(index=index, variable=variable, exp=exp, m=m, ref_period=ref_period, study_period=study_period)
                                print (f'{m} finished...\n')                      

            print ("*** ALL MODELS COMPUTED ***")
        except Exception:
            traceback.print_exc(file=log)
            pass