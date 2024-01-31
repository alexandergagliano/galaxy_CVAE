import numpy as np
import h5py 
import glob 

# Assuming 'hfile_path' is the path to your HDF5 file
train_data_path = '/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/TrainSample/segmented/*.h5'
val_data_path = '/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/TestSample/segmented/*.h5'

for filepath in [train_data_path, val_data_path]:
    for hfile_path in glob.glob(filepath):
        with h5py.File(hfile_path, 'r+') as hfile:
            photoz = hfile['photoz'][:]
            mass = hfile['mass'][:]
            sfr = hfile['sfr'][:]
            photoz_err = hfile['photoz_err'][:]
            mass_err = hfile['mass_err'][:]
            sfr_err = hfile['sfr_err'][:]

            # Stack once and save
            labels = np.vstack([photoz, mass, sfr, photoz_err, mass_err, sfr_err]).T.astype("float32")

            # Save the precomputed labels to the HDF5 file or as a separate file
            if 'labels' not in hfile:
                hfile.create_dataset('labels', data=labels)
