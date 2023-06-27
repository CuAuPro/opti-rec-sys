import pandas as pd
import numpy as np
from scipy.optimize import minimize
import torch
from modelling.preprocessing import get_bin_labels
from .utils import most_similar_index
from typing import List


"""
# Script for finding historical-based limits of recipe parameters
limits = []
for pass_nr in range(1,int(max(raw_df.pass_nr)+1)):
  limits_pass = []
  for col in recipe_columns:
    limits_pass_col = [raw_df.loc[raw_df.pass_nr == pass_nr][col].min(),raw_df.loc[raw_df.pass_nr == pass_nr][col].max()]
    limits_pass_col = [ round(elem, 2) for elem in limits_pass_col ]
    limits_pass.append(limits_pass_col)
  limits.append(limits_pass)   
"""
class Optimizer():
    def __init__(self, model, pass_nr:int, dh_tol:float, sc_recipe, recipe_dict:dict, categorizers, bounds:List, lambda1:float=0.5, method:str='Powell'):
        """Optimization object

        Args:
            model (_type_): PyTorch model
            pass_nr (int): pass number (necessary for bounds setting)
            dh_tol (float): Thicknes deviation tolerance
            sc_recipe (_type_): scaler object for recipe columns
            recipe_dict (dict): recipe id lookup table
            categorizers (_type_): categorizer/binarizer object
            bounds (List): bounds for tensions and speed
            lambda1 (float, optional): Optimization parameter (Higher value give more importance on quality). Defaults to 0.5.
            method (str, optional): Optimization method. Defaults to 'Powell'.
        """
        self.bounds = np.array(bounds)

        self.method = method
        self.lambda1 = lambda1
        self.dh_tol = dh_tol #um
        self.model = model
        self.pass_nr = pass_nr
        self.sc_recipe = sc_recipe
        self.recipe_dict = recipe_dict
        self.categorizers = categorizers

    def optimize(self, x0: np.array, x_args: torch.Tensor, coil_index: torch.Tensor, recipe_index: torch.Tensor, bounds: np.array=None):
        """Optimization wrapper

        Args:
            x0 (np.array): Initial recipe
            x_args (torch.Tensor): Additional static (coil) features
            coil_index (torch.Tensor): Index of coil
            recipe_index (torch.Tensor): Index of recipe
            bounds (np.array, optional): Tight bounds

        Returns:
            _type_: Optimization result
        """
        if bounds is None:
            bounds = self.bounds[self.pass_nr]
            
        return minimize(self.opt_fcn_pred, x0, args=[coil_index, recipe_index, x_args, self.sc_recipe], bounds=bounds, method=self.method)

    def opt_fcn_pred(self, x_in, params):
            self.lambda2 = 1-self.lambda1
            coil_idx = params[0]
            coil_features = params[2]
            recipe_columns = ['tension_en', 'tension_ex', 'velocity_mdr']
            x_in = pd.DataFrame(x_in.reshape(1,-1), columns=recipe_columns)
            x_in_bin = x_in.copy()
            for column in recipe_columns:
              x_in_bin[column] =  get_bin_labels(x_in[column].values, self.categorizers[column], padding=2)
            #print(x_in)
            recipe_idx = x_in_bin[recipe_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
            recipe_idx = recipe_idx.values
            del x_in_bin

            if not(recipe_idx in list(self.recipe_dict.keys())):
              #print("Finding similar of: {}".format(recipe_idx))
              recipe_idx = most_similar_index(recipe_idx[0], list(self.recipe_dict.keys()))
              recipe_idx = [recipe_idx] # Convert to suitable type for transforming in tensor
              #print("Found similar: {}".format(recipe_idx))

            recipe_idx = ([self.recipe_dict[pid] for pid in recipe_idx])
            #print(recipe_idx)
            recipe_idx = torch.Tensor(recipe_idx).to(torch.int32)
            #print(recipe_idx)
            x_in = torch.Tensor(self.sc_recipe.transform(x_in)).to(torch.float32)
            y = self.model(coil_idx, recipe_idx, coil_features, x_in).detach().numpy()
            J1 = y
            #print(J1)
            new_speed = self.sc_recipe.inverse_transform(abs(x_in.detach().numpy())).flatten()[-1]
            J2 = (max(self.bounds[self.pass_nr][2]) - new_speed)
            J1 = J1/np.linalg.norm(self.dh_tol) #normalize to tolerance
            J2 = J2/max(self.bounds[self.pass_nr][2])
            #print(self.lambda2*(J2**2))

            J3 = sum(y[y>self.dh_tol] - self.dh_tol)

            #print([J1,J2,J3])
            Jtot = self.lambda1*(J1**2) + self.lambda2*(J2**2) + (J3**2)

            return Jtot

    