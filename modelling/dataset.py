import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List

class ZRM_ReccomenderDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 coil_features: List[str],
                 recipe_features: List[str],
                 quality_feature: str):
        """ZRM Reccomender System Dataset object

        Args:
            data (pd.DataFrame): Dataset
            coil_features (List[str]): List of coil feature names
            recipe_features (List[str]): List of recipe feature names
            quality_feature (str): Quality feature name
        """

        self.coil_indices = torch.FloatTensor(data['coil_idx'].values).to(torch.int32)
        self.recipe_indices = torch.FloatTensor(data['recipe_idx'].values).to(torch.int32)
        self.coil_features =  torch.FloatTensor(data[coil_features].values).to(torch.float32)
        self.recipe_features = torch.FloatTensor(data[recipe_features].values).to(torch.float32)
        self.quality = torch.FloatTensor(data[quality_feature].values).to(torch.float32)

    def __len__(self):
        return len(self.coil_indices)
    
    def __getitem__(self, idx):
        return (self.coil_indices[idx], self.recipe_indices[idx], self.coil_features[idx], 
                self.recipe_features[idx], self.quality[idx])