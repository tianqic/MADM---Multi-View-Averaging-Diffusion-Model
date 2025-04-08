import os
import glob
import numpy as np
#from utility.utility import minmax_normalization
from torch.utils.data import Dataset
import nibabel as nib
import random
from tqdm import tqdm
from skimage.transform import rescale
import torch
from torch.utils.data import DataLoader
import scipy.ndimage as ndimage

def normalize_ac(data, mask):
    non_zero_data = data[mask == True]
    non_zero_mean = np.mean(non_zero_data)
    data = data / non_zero_mean
    data = np.tanh(data / 5)
    return data


def get_mask(nac, num_blurred = 5, threshold = 0.05):
    blurred_nac = np.copy(nac)
    for i in range(num_blurred):
        blurred_nac = ndimage.gaussian_filter(blurred_nac, sigma=1)
    mask = np.where(nac > threshold, True, False)
    return mask


class LoadPetSlices(Dataset):
    def __init__(self, root_dir = r"", axis = "x", load_adj = 8, seed = 1, out_size = 192) -> None:
        super().__init__()
        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir
        self.file_names = os.listdir(os.path.join(root_dir, '5NAC'))
        self.ids = [i[:len(i) - 9] for i in self.file_names]

        print(self.ids)

        self.len = len(self.ids)
        self.ac_data = []
        self.nac_data = []

        for i in tqdm(range(len(self.ids))):
            nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
            ac_path = os.path.join(self.root_dir, '100AC', self.ids[i] + "100_AC.nii")
            nac = nib.load(nac_path).get_fdata()
            ac = nib.load(ac_path).get_fdata()
            mask = get_mask(nac)
            self.ac_data.append(self.process_nii_file(ac, mask))
            self.nac_data.append(self.process_nii_file(nac, mask))
            


    def __len__(self):
        return len(self.ids)
    
    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z = 0):
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1,0,2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2,0,1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)
        comb_slices[fill_start:fill_start + slices.shape[0]] = slices

        return comb_slices
    


    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)
        shape=(self.out_size, self.out_size, max(data.shape[2], self.out_size))
        new_data = np.zeros(shape)
        new_data[:,:,:data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2, (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2, :]
        return torch.from_numpy(new_data)
    
    def random_flip(self, image, label, flip_axis, p = 0.3):
        for i in flip_axis:
            if np.random.rand() < p:
                image = torch.flip(image, [i])
                label = torch.flip(label, [i])
        return image, label

    def __getitem__(self, idx):
        ac = self.ac_data[idx]
        nac = self.nac_data[idx]

        z_size = ac.shape[2]
        if self.axis != "z" and z_size > self.out_size:
            start_z = random.randint(0, z_size - self.out_size)
            central_slice_idx = random.randint(0, self.out_size - 1) 
        else:
            start_z = 0
            central_slice_idx = random.randint(0, z_size - 1) 
        

        
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)

        if self.axis == "x":
            ac = ac[central_slice_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            ac = ac[:, central_slice_idx, start_z:start_z + self.out_size]
        elif self.axis == "z":
            ac = ac[:, :, central_slice_idx]
        ac = ac.unsqueeze(0)

        nac, ac = self.random_flip(nac, ac, flip_axis=[0,1,2])
        return ac, nac

def load_data(batch_size, root_dir = r"", axis = "x", load_adj = 8, seed = 1, out_size = 192):
    dataset = LoadPetSlices(root_dir = root_dir, axis = axis, load_adj = load_adj, seed = seed, out_size = out_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        yield from loader


class LoadTestData():
    def __init__(self, root_dir = r"", axis = "z", load_adj = 8, seed = 1, out_size = 192) -> None:
        
        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir
        self.file_names = os.listdir(os.path.join(root_dir, '5NAC'))
        self.ids = [i[:len(i) - 9] for i in self.file_names]
    
        self.len = len(self.ids)
        self.nac_data = []
        self.original_z = []
        
        for i in tqdm(range(len(self.ids))):
            nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
            nac = nib.load(nac_path).get_fdata()
            mask = get_mask(nac)
            self.original_z.append(nac.shape[2])
            nac = self.process_nii_file(nac, mask)
            self.nac_data.append(nac)
    
        self.idx = None
    
    def __len__(self):
        return self.len
    
    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z = 0):
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1,0,2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2,0,1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)
        comb_slices[fill_start:fill_start + slices.shape[0]] = slices

        return comb_slices

    def get_zsize(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.nac_data[idx].shape[2]

    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)
        shape=(self.out_size, self.out_size, max(data.shape[2], self.out_size))
        new_data = np.zeros(shape)
        new_data[:,:,:data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2, (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2, :]
        return torch.from_numpy(new_data)

    

    def get_slices(self, central_slice_idx, idx = None, start_z = 0):
        if idx == None:
            idx = self.idx
        nac = self.nac_data[idx]
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)
        return nac
    
    def get_target(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.ac[idx]
    
    def get_name(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.ids[idx]
    
    def get_original_z(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.original_z[idx]
        
    def normalize(self, data, idx = None):
        if idx == None:
            idx = self.idx
        nac_mean = self.means[idx]
        data /= nac_mean
        data = np.tanh(data / 5)
        return data

class LoadValData():
    def __init__(self, root_dir = r"", axis = "z", load_adj = 8, seed = 1, out_size = 192) -> None:
        
        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir
        self.file_names = os.listdir(os.path.join(root_dir, '5NAC'))
        self.ids = [i[:len(i) - 9] for i in self.file_names]
    
        self.len = len(self.ids)
        self.original_z = []

        self.ac_data = []
        self.nac_data = []

        for i in tqdm(range(len(self.ids))):
            nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
            ac_path = os.path.join(self.root_dir, '100AC', self.ids[i] + "100_AC.nii")
            nac = nib.load(nac_path).get_fdata()
            ac = nib.load(ac_path).get_fdata()
            mask = get_mask(nac)
            self.original_z.append(nac.shape[2])
            ac = self.process_nii_file(ac, mask)
            self.ac_data.append(ac)
            nac = self.process_nii_file(nac, mask)
            self.nac_data.append(nac)
            
            
        self.idx = None
    
    def __len__(self):
        return self.len
    
    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z = 0):
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1,0,2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2,0,1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)
        comb_slices[fill_start:fill_start + slices.shape[0]] = slices

        return comb_slices

    def get_zsize(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.nac_data[idx].shape[2]

    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)
        shape=(self.out_size, self.out_size, max(data.shape[2], self.out_size))
        new_data = np.zeros(shape)
        new_data[:,:,:data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2, (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2, :]
        return torch.from_numpy(new_data)
    

    def get_slices(self, central_slice_idx, idx = None, start_z = 0):
        if idx == None:
            idx = self.idx
        nac = self.nac_data[idx]
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)
        return nac

    def get_target_slices(self, central_slice_idx, idx = None, start_z = 0):
        if idx == None:
            idx = self.idx
        ac = self.ac_data[idx]
        if self.axis == "x":
            ac = ac[central_slice_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            ac = ac[:, central_slice_idx, start_z:start_z + self.out_size]
        elif self.axis == "z":
            ac = ac[:, :, central_slice_idx + start_z]
        ac = ac.unsqueeze(0)
        return ac


    def get_target(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.ac[idx]
    
    def get_name(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.ids[idx]
    
    def get_original_z(self, idx = None):
        if idx == None:
            idx = self.idx
        return self.original_z[idx]
    