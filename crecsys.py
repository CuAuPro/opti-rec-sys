# Import NumPy, pandas, matplotlib and PyTorch
import numpy as np
import pandas as pd

# Import PyTorch
import torch

# Import sklearn modules
from sklearn.metrics.pairwise import cosine_similarity

# Import Utility Functions
import pickle
import json

# Import custom modules
#import engine.settings as settings
#from engine import settings as settings
import engine.settings as settings

#from modelling.ncf import ZRM_Reccomender
from modelling.preprocessing import feature_engineering, get_bin_labels
from engine.optimizer import Optimizer
import engine.optimizer_adv as optimizer_adv
from engine.utils import most_similar_index, importConfig, calculateCriterion, calc_dh_tol
from ZRMlib.profile_generator import trapezoidal_profile

from sklearn.cluster import DBSCAN

from scipy import stats

##################################################
#               PARAMETERS
simi_thresh_similar_coils = 0.9 # Used for selection of simi_thresh_similar_coils recipes
N_min_similar_coil_classes = 5  # Used for selection of N_similar_coils (meaning binarized coil classes)
                                # recipes if there are less than
                                # N_similar_coils, when simi_thresh_similar_coils used
                     
N_best_recipes = 7          # Used for selection of best performed N_best_recipes recipes
N_best_recipes_clust = 20
opt_lambda1_def = 0.5       # Optimization parameter (Higher value give more importance on quality).
opt_method_def = 'Powell'   # Optimization method
dh_tol_def = 6.0
##################################################

settings.init()
out = importConfig("engine/config.json")
device = torch.device("cpu")
#print("Inference on CPU")

# Load preprocessed data
df = pd.read_pickle(settings.config["engine"]["data"])
df = df.reset_index()

# Load model
model = torch.load(settings.config["engine"]["model"],map_location=device)
model.eval()
model.to(device)
# Load categorizers
with open(settings.config["engine"]["categorizers"], 'rb') as f:
  categorizers = pickle.load(f)

# Load scalers
with open(settings.config["engine"]["sc_coil"], "rb") as f:
  sc_coil = pickle.load(f)
with open(settings.config["engine"]["sc_recipe"], "rb") as f:
  sc_recipe = pickle.load(f)

# Load lookup table for coil/recipe indices
with open(settings.config["engine"]["coil_dict"], "rb") as f:
  coil_dict = pickle.load(f)
with open(settings.config["engine"]["recipe_dict"], "rb") as f:
  recipe_dict = pickle.load(f)

# Calculate coil/recipe (user/item) matrix
crm = df.pivot_table(values='EXIT_THICK_DEVIATION_ABS_AVG', index='coil_idx', columns='recipe_idx', aggfunc='mean').fillna(0)
simi_coil = cosine_similarity(crm)

padding=2
coil_bins = pd.DataFrame(coil_dict.keys(), columns=['Column'])
coil_bins = pd.concat([coil_bins['Column'].str[i:i+padding] for i in range(0, len(coil_bins['Column'][0]), padding)], axis=1)
coil_bins.columns = [ 'h_entry',
                      'h_entry_std',
                      'h_entry_min',
                      'h_entry_max',
                      'h_exit_ref',
                      'h_reduced',
                      'h_reduction',
                      'AL',
                      'CU'
                      ] 
coil_bins = coil_bins.astype(int)

class ZRMrecsys():

    # data to be passed into model
    feature_names = ['dh_entry', 'dh_entry_min', 'dh_entry_max', 'dh_entry_std',
                    'hprint(self_entry_ref','h_exit_ref','REF_INITIAL_THICKNESS', 'dh_profile']
    
    coil_columns  = [ 'h_entry',
                      'h_entry_std',
                      'h_entry_min',
                      'h_entry_max',
                      'h_exit_ref',
                      'h_reduced',
                      'h_reduction',
                      'AL',
                      'CU'
                      ] 
    recipe_columns = ['tension_en',
                      'tension_ex',
                      'velocity_mdr'
                     ]
    quality_column =  'EXIT_THICK_DEVIATION_ABS_AVG'

    
    
    def __init__(self, infer_type=None):
        self.bounds = settings.config["recipe_bounds"]
        if infer_type == None:            
            self._infer_type = 'optimise'
            
        elif any(infer_type == a for a in ['optimise', 'optimise_adv', 'recommend']):
            self._infer_type = infer_type
        
        return
    def optimize(self, coil_df:pd.DataFrame, pass_nr:int, dh_tol:float=dh_tol_def, opt_lambda1:float=opt_lambda1_def, opt_method:str=opt_method_def):
        res1 = recsys_obj.optimize_1(coil_df, pass_nr, dh_tol, opt_lambda1, opt_method)
        res1['Jtot'] = calculateCriterion(res1["velocity_mdr"].values, res1["pred"].values,
                                               self.bounds[pass_nr], dh_tol=dh_tol, lambda1=opt_lambda1)
        
        res2 = recsys_obj.optimize_2(coil_df, pass_nr, dh_tol, opt_lambda1, opt_method)
        res2['Jtot'] = calculateCriterion(res2["velocity_mdr"].values, res2["pred"].values,
                                               self.bounds[pass_nr], dh_tol=dh_tol, lambda1=opt_lambda1)
        
        # Compare the Jtot values
        if res1['Jtot'].iloc[0] < res2['Jtot'].iloc[0]:
            res = res1
        else:
            res = res2
        res = res.drop(columns='Jtot')
    
    def optimize_1(self, coil_df:pd.DataFrame, pass_nr:int, dh_tol:float=dh_tol_def, opt_lambda1:float=opt_lambda1_def, opt_method:str=opt_method_def):
        """Model based recipe optimization


        Args:
            coil_df (pd.DataFrame): One row DataFrame with coil data
            pass_nr (int): Pass number (necessary for bounds setting)
            dh_tol (float, optional): Thickness deviation tolerance. Defaults to dh_tol_def.
            opt_lambda1 (float, optional): Optimization parameter (Higher value give more importance on quality). Defaults to 0.5. Defaults to opt_lambda1_def.
            opt_method (str, optional): Optimization method. Defaults to opt_method_def.

        Returns:
            pd.DataFrame: DataFrame with recipe and predicted quality
        """


        final_res, new_coil, new_coil_idx, new_recipe_idx = self.recommend(coil_df, int(pass_nr), dh_tol, opt_lambda1)

        # prepare optimizer
        opt = Optimizer(model, int(pass_nr), dh_tol, sc_recipe, recipe_dict, categorizers, self.bounds, lambda1=opt_lambda1,
                        method=opt_method)
        x0 = final_res[self.recipe_columns].values

        # calculate moroe tight bounds (based on recipe_idx bins)
        bounds = self.tight_bounds(new_recipe_idx[0], pass_nr)
        bounds=None
        # prepare new_recipe_idx
        x_args = torch.Tensor(sc_coil.transform(new_coil[self.coil_columns])).to(torch.float32)
        new_coil_idx = torch.Tensor(new_coil_idx).to(torch.int32)
        new_recipe_idx = torch.Tensor(new_recipe_idx).to(torch.int32)
        
        # run optimization
        res = opt.optimize(x0, x_args, new_coil_idx, new_recipe_idx, bounds)
        final_res_opt = pd.DataFrame(data=res.x.reshape(1, -1), columns=self.recipe_columns)
        final_res_opt['pred'] = self.predictQuality(final_res_opt, new_coil, new_coil_idx)

        return final_res_opt

    def optimize_2(self, coil_df:pd.DataFrame, pass_nr:int, dh_tol:float=dh_tol_def, opt_lambda1:float=opt_lambda1_def, opt_method:str=opt_method_def):
        """Model based recipe optimization - pymoo optimization library


        Args:
            coil_df (pd.DataFrame): One row DataFrame with coil data
            pass_nr (int): Pass number (necessary for bounds setting)
            dh_tol (float, optional): Thickness deviation tolerance. Defaults to dh_tol_def.
            opt_lambda1 (float, optional): Optimization parameter (Higher value give more importance on quality). Defaults to 0.5. Defaults to opt_lambda1_def.
            opt_method (str, optional): Optimization method. Defaults to opt_method_def.

        Returns:
            pd.DataFrame: DataFrame with recipe and predicted quality
        """

        final_res, new_coil, new_coil_idx, new_recipe_idx = self.recommend(coil_df, int(pass_nr), dh_tol, opt_lambda1)

        # prepare optimizer
        opt = optimizer_adv.Optimizer(model, int(pass_nr), dh_tol, sc_recipe, recipe_dict, categorizers, self.bounds, lambda1=opt_lambda1,
                        method=opt_method)
        x0 = final_res[self.recipe_columns].values

        # calculate moroe tight bounds (based on recipe_idx bins)
        bounds = self.tight_bounds(new_recipe_idx[0], pass_nr)
        bounds=None
        #print(bounds)
        # prepare new_recipe_idx
        x_args = torch.Tensor(sc_coil.transform(new_coil[self.coil_columns])).to(torch.float32)
        new_coil_idx = torch.Tensor(new_coil_idx).to(torch.int32)
        new_recipe_idx = torch.Tensor(new_recipe_idx).to(torch.int32)

        # run optimization
        res = opt.optimize(x0, x_args, new_coil_idx, new_recipe_idx, bounds)
        final_res_opt = pd.DataFrame(data=res.X.reshape(1, -1), columns=self.recipe_columns)
        
        final_res_opt['pred'] = self.predictQuality(final_res_opt, new_coil, new_coil_idx)

        return final_res_opt

    def recommend(self, coil_df:pd.DataFrame, pass_nr:int, dh_tol:float=dh_tol_def, opt_lambda1:float=opt_lambda1_def):
        """Recommend recipe settings based on coil similarity

        Args:
            coil_df (pd.DataFrame): One row DataFrame with coil data
            pass_nr (int): Pass number (necessary for bounds setting)
            dh_tol (float, optional): Thickness deviation tolerance. Defaults to dh_tol_def.
            opt_lambda1 (float, optional): Optimization parameter (Higher value give more importance on quality). Defaults to opt_lambda1_def.

        Returns:
            _type_: _description_
        """
        
        new_coil = feature_engineering(coil_df)
        new_coil = new_coil[['pass_nr'] + self.coil_columns]
        
        # Binarize data
        new_coil_bin = pd.DataFrame()
        for column in self.coil_columns:
          new_coil_bin[column]  = get_bin_labels(new_coil[column].values, categorizers[column], padding=2)
        coil_idx = new_coil_bin[self.coil_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        coil_idx = coil_idx.values
        new_coil_bin = new_coil_bin.astype(int)

        # Find similar coil if necessary
        if not (coil_idx in list(coil_dict.keys())):
            print("Finding similar of: {}".format(coil_idx))
            coil_idx = most_similar_index(coil_idx[0], list(coil_dict.keys()))
            coil_idx = [coil_idx]  # Convert to suitable type for transforming in tensor
            print("Found similar: {}".format(coil_idx))

        new_coil.loc[:,'coil_idx'] = ([coil_dict[pid] for pid in coil_idx])
        new_coil_idx = [int(new_coil['coil_idx'])]
        
        #print('new_coil_idx')
        #print(new_coil_idx)
        """
        # Select most similar coils
        #simi_idxs = np.argsort(simi.flatten())[::-1]
        #simi_idxs = simi_idxs[0:N_similar_coils]
        new_coil_simi = simi_coil[new_coil_idx]
        simi_idxs = np.argsort(new_coil_simi)
        simi_idxs = simi_idxs[new_coil_simi[simi_idxs] > simi_thresh_similar_coils][::-1]
        if simi_idxs.shape[0] < N_min_similar_coil_classes:
            simi_idxs = np.argsort(new_coil_simi)
            simi_idxs = simi_idxs[::-1][0:N_min_similar_coil_classes]

        most_similar_coils = crm.index[simi_idxs]

        crm_coil = crm.loc[most_similar_coils]
        #print(most_similar_coils)

        # Melt the user-item matrix to transform it back into a DataFrame format
        final_res = pd.melt(crm_coil.reset_index(), id_vars=['coil_idx'], var_name='recipe_idx', value_name=self.quality_column)
        # Extract only those values where quality_column is not 0 (none of coil used that recipe)
        final_res = final_res.loc[final_res[self.quality_column] != 0]
        final_res = final_res[['coil_idx', 'recipe_idx']]
        final_res = pd.merge(final_res, df, on=['coil_idx', 'recipe_idx'], how='inner')
        """
        user_similarities = cosine_similarity(new_coil_bin[self.coil_columns], coil_bins)[0]
        most_similar_idx = np.argsort(user_similarities)[::-1]

        most_similar_idx = most_similar_idx[user_similarities[most_similar_idx] > simi_thresh_similar_coils][:N_min_similar_coil_classes]
        if most_similar_idx.shape[0] < N_min_similar_coil_classes:
          most_similar_idx = np.argsort(user_similarities)[::-1]
          most_similar_idx = most_similar_idx[:N_min_similar_coil_classes]
        
        final_res = df.loc[df['coil_idx'].isin(most_similar_idx)]
        # Extract only those values where quality_column is not 0 (none of coil used that recipe)
        final_res = final_res.loc[final_res[self.quality_column] != 0]

        final_res["Jtot"] = calculateCriterion(final_res["velocity_mdr"].values, final_res[self.quality_column].values,
                                               self.bounds[pass_nr], dh_tol=dh_tol, lambda1=opt_lambda1)
        idxs_best = np.argsort(final_res["Jtot"].values)

        final_res = final_res.iloc[idxs_best].iloc[0:N_best_recipes]
        #new_coil_idx, new_recipe_idx = final_res[['coil_idx', 'recipe_idx']].median().values
        #new_coil_idx, new_recipe_idx = stats.mode(final_res[['coil_idx', 'recipe_idx']].values, keepdims=False).mode
        #new_coil_idx = new_coil_idx.flatten()
        #new_recipe_idx = new_recipe_idx.flatten()
        #final_res = pd.DataFrame(final_res[self.recipe_columns].median()).T
        weights = 1-final_res["Jtot"].values
        final_res = pd.DataFrame((final_res[self.recipe_columns].T @ weights) / np.sum(weights)).T
        
        
        new_recipe_bin = pd.DataFrame()
        for column in self.recipe_columns:
          new_recipe_bin[column]  = get_bin_labels(final_res[column].values, categorizers[column], padding=2)
        new_recipe_idx = new_recipe_bin[self.recipe_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        new_recipe_idx = new_recipe_idx.values
        # Find similar coil if necessary
        if not (new_recipe_idx in list(recipe_dict.keys())):
            print("Finding similar of: {}".format(new_recipe_idx))
            new_recipe_idx = most_similar_index(new_recipe_idx[0], list(coil_dict.keys()))
            new_recipe_idx = [new_recipe_idx]  # Convert to suitable type for transforming in tensor
            print("Found similar: {}".format(new_recipe_idx))
        new_recipe_idx = ([recipe_dict[pid] for pid in new_recipe_idx])

        final_res['pred'] = self.predictQuality(final_res, new_coil, new_coil_idx)

        return final_res, new_coil, new_coil_idx, new_recipe_idx

    def recommendClust(self, coil_df:pd.DataFrame, pass_nr:int, dh_tol:float=dh_tol_def, opt_lambda1:float=opt_lambda1_def):
        """Recommend recipe settings based on coil similarity

        Args:
            coil_df (pd.DataFrame): One row DataFrame with coil data
            pass_nr (int): Pass number (necessary for bounds setting)
            dh_tol (float, optional): Thickness deviation tolerance. Defaults to dh_tol_def.
            opt_lambda1 (float, optional): Optimization parameter (Higher value give more importance on quality). Defaults to opt_lambda1_def.

        Returns:
            _type_: _description_
        """
        
        new_coil = feature_engineering(coil_df)
        new_coil = new_coil[['pass_nr'] + self.coil_columns]
        
        # Binarize data
        new_coil_bin = pd.DataFrame()
        for column in self.coil_columns:
          new_coil_bin[column]  = get_bin_labels(new_coil[column].values, categorizers[column], padding=2)
        coil_idx = new_coil_bin[self.coil_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        coil_idx = coil_idx.values
        new_coil_bin = new_coil_bin.astype(int)

        # Find similar coil if necessary
        if not (coil_idx in list(coil_dict.keys())):
            print("Finding similar of: {}".format(coil_idx))
            coil_idx = most_similar_index(coil_idx[0], list(coil_dict.keys()))
            coil_idx = [coil_idx]  # Convert to suitable type for transforming in tensor
            print("Found similar: {}".format(coil_idx))

        new_coil = new_coil.reset_index()
        new_coil.loc[:,'coil_idx'] = ([coil_dict[pid] for pid in coil_idx])
        new_coil_idx = [int(new_coil['coil_idx'])]
        
        user_similarities = cosine_similarity(new_coil_bin[self.coil_columns], coil_bins)[0]
        most_similar_idx = np.argsort(user_similarities)[::-1]

        most_similar_idx = most_similar_idx[user_similarities[most_similar_idx] > simi_thresh_similar_coils][:N_min_similar_coil_classes]
        if most_similar_idx.shape[0] < N_min_similar_coil_classes:
          most_similar_idx = np.argsort(user_similarities)[::-1]
          most_similar_idx = most_similar_idx[:N_min_similar_coil_classes]
        
        final_res = df.loc[df['coil_idx'].isin(most_similar_idx)]
        # Extract only those values where quality_column is not 0 (none of coil used that recipe)
        final_res = final_res.loc[final_res[self.quality_column] != 0]

        final_res["Jtot"] = calculateCriterion(final_res["velocity_mdr"].values, final_res[self.quality_column].values,
                                               self.bounds[pass_nr], dh_tol=dh_tol, lambda1=opt_lambda1)
        #idxs_best = np.argsort(final_res["Jtot"].values)

        final_res = final_res.sort_values('Jtot')[self.recipe_columns+['Jtot']].copy()
        final_res = final_res.iloc[0:N_best_recipes_clust]

        # Normalize the selected columns using the min-max scaling
        selection = (final_res[self.recipe_columns] - final_res[self.recipe_columns].min()) / (final_res[self.recipe_columns].max() - final_res[self.recipe_columns].min())
        selection['Jtot'] = final_res['Jtot']
        
        # Initialize DBSCAN
        dbscan = DBSCAN(eps=0.15, min_samples=3)

        # Perform clustering
        final_res['cluster'] = dbscan.fit_predict(selection[self.recipe_columns])
        final_res = final_res.loc[final_res['cluster'] != -1] # We do not want outliers
        final_res = final_res.groupby('cluster').mean()
        final_res = final_res.sort_values('Jtot')
        #print(final_res)
        #print("Selected cluster: {}".format(final_res.index[0]))
        final_res = final_res.iloc[[0]].reset_index(drop=True)
        final_res = final_res[self.recipe_columns]
                
        new_recipe_bin = pd.DataFrame()
        for column in self.recipe_columns:
          new_recipe_bin[column]  = get_bin_labels(final_res[column].values, categorizers[column], padding=2)
        new_recipe_idx = new_recipe_bin[self.recipe_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        new_recipe_idx = new_recipe_idx.values
        # Find similar coil if necessary
        if not (new_recipe_idx in list(recipe_dict.keys())):
            print("Finding similar of: {}".format(new_recipe_idx))
            new_recipe_idx = most_similar_index(new_recipe_idx[0], list(coil_dict.keys()))
            new_recipe_idx = [new_recipe_idx]  # Convert to suitable type for transforming in tensor
            print("Found similar: {}".format(new_recipe_idx))
        new_recipe_idx = ([recipe_dict[pid] for pid in new_recipe_idx])

        final_res['pred'] = self.predictQuality(final_res[self.recipe_columns], new_coil, new_coil_idx)

        return final_res, new_coil, new_coil_idx, new_recipe_idx

    def get_recipe(self, coil_df, pass_nr, dh_tol, opt_lambda1=opt_lambda1_def):
        
        coil_df.loc[:,'pass_nr'] = pass_nr
        coil_df.loc[:,'coil_id'] = '00112233'

        if len(coil_df) > 1: 
            coil_df = self.get_steady_state_features(coil_df, pass_nr)

        if self._infer_type == 'optimise' or self._infer_type=='optimize':
            opt_res = self.optimize(coil_df, pass_nr, dh_tol, opt_lambda1)

        elif self._infer_type == 'optimise_1' or self._infer_type == 'optimize_1':
            opt_res = self.optimize_1(coil_df, pass_nr, dh_tol, opt_lambda1)
            
        elif self._infer_type == 'optimise_2' or self._infer_type == 'optimize_2':
            opt_res = self.optimize_2(coil_df, pass_nr, dh_tol, opt_lambda1)
            
        elif self._infer_type == 'recommend':
            opt_res = self.recommend(coil_df, pass_nr, dh_tol, opt_lambda1)
            opt_res = opt_res[0]
            
        return opt_res

            
    def predictQuality(self, x_suggested, new_coil, coil_idx):
        # prepare new_recipe_idx
        x_args = torch.Tensor(sc_coil.transform(new_coil[self.coil_columns])).to(torch.float32)
        coil_idx = torch.Tensor(coil_idx).to(torch.int32)
        x_suggested_bin = x_suggested.copy()
        for column in self.recipe_columns:
          x_suggested_bin[column] = get_bin_labels(x_suggested[column].values, categorizers[column], padding=2)
        recipe_idx = x_suggested_bin[self.recipe_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        del x_suggested_bin
        recipe_idx = recipe_idx.values

        if not(recipe_idx in list(recipe_dict.keys())):
          print("Finding similar of: {}".format(recipe_idx))
          recipe_idx = most_similar_index(recipe_idx[0], list(recipe_dict.keys()))
          recipe_idx = [recipe_idx] # Convert to suitable type for transforming in tensor
          print("Found similar: {}".format(recipe_idx))

        recipe_idx = ([recipe_dict[pid] for pid in recipe_idx])
        recipe_idx = torch.Tensor(recipe_idx).to(torch.int32)
        x_suggested = torch.Tensor(sc_recipe.transform(x_suggested)).to(torch.float32)

        y = model(coil_idx, recipe_idx, x_args, x_suggested).detach().numpy()

        return y
    
    def preprocess_coil_data(self, coil_df):
        new_coil = feature_engineering(coil_df)
        new_coil = new_coil[['pass_nr'] + self.coil_columns]
        
        # Binarize data
        new_coil_bin = new_coil.copy()
        for column in self.coil_columns:
            new_coil_bin[column] = get_bin_labels(new_coil_bin[column].values, categorizers[column], padding=2)

        coil_idx = new_coil_bin[self.coil_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        coil_idx = coil_idx.values
        # Find similar coil if necessary
        if not (coil_idx in list(coil_dict.keys())):
            #print("Finding similar of: {}".format(coil_idx))
            coil_idx = most_similar_index(coil_idx[0], list(coil_dict.keys()))
            coil_idx = [coil_idx]  # Convert to suitable type for transforming in tensor
            #print("Found similar: {}".format(coil_idx))
        #print(coil_idx)

        new_coil.loc[:,'coil_idx'] = ([coil_dict[pid] for pid in coil_idx])
        new_coil_idx = [int(new_coil['coil_idx'])]

        return new_coil, new_coil_idx
    
    def get_steady_state_features(self, coil_df, pass_nr):
        
        dh_entry_profile = np.array(coil_df["dh_profile"]).reshape(-1,1)
        coil_length = coil_df["dh_profile"].index[-1] # Index must be length at current sample
        # We need only one row now
        coil_df = coil_df.iloc[[0]].copy()
        if pass_nr > 1:
            # because in pass_nr=1 division by zero can occur
            coil_length = coil_length*coil_df["REF_INITIAL_THICKNESS"].iloc[0]/ (coil_df["REF_INITIAL_THICKNESS"].iloc[0]-coil_df["h_entry_ref"].iloc[0])

        max_speed = self.bounds[pass_nr][2][1]/60
        t,v,p, d_acc, d_const, d_dec  = trapezoidal_profile(coil_length, max_speed, 0.4, nr_samples = dh_entry_profile.shape[0], return_segment_distances=True)
        idx_const_start = (np.abs(p - d_acc)).argmin()
        idx_const_end = (np.abs(p - (d_acc + d_const))).argmin()
        dh_entry_profile = dh_entry_profile[idx_const_start:idx_const_end]

        coil_df['dh_entry'] = np.mean(dh_entry_profile)
        coil_df['dh_entry_min'] = np.min(dh_entry_profile)
        coil_df['dh_entry_max'] = np.max(dh_entry_profile)
        coil_df['dh_entry_std'] = np.std(dh_entry_profile, ddof=1)
        coil_df = coil_df.iloc[[0]]
        return coil_df

    
    def tight_bounds(self, recipe_idx, pass_nr):

      recipe_sign = list(recipe_dict.keys())[int(recipe_idx)]
      recipe_idxs = np.array([int(recipe_sign[i:i+2]) for i in range(0, len(recipe_sign), 2)])
      bounds = []

      for i, col in enumerate(self.recipe_columns):
        bin_idx = recipe_idxs[i]
        bottom_limit = categorizers[col][bin_idx].left
        upper_limit = categorizers[col][bin_idx].right
        if bottom_limit < self.bounds[pass_nr][i][0]:
          bottom_limit = self.bounds[pass_nr][i][0]
        if bottom_limit > self.bounds[pass_nr][i][1]:
          bottom_limit = self.bounds[pass_nr][i][0]

        if upper_limit > self.bounds[pass_nr][i][1]:
          upper_limit = self.bounds[pass_nr][i][1]
        if upper_limit < self.bounds[pass_nr][i][0]:
          upper_limit = self.bounds[pass_nr][i][1]
        bounds_col = [bottom_limit, upper_limit]
        bounds.append(bounds_col)
      
      bounds = np.array(bounds)
      return bounds

if __name__ =='__main__':
    with open("example_call_json.json") as f:
        data_json = json.load(f)
    data_json = data_json["1"]
    df_json = pd.DataFrame(data_json["x_var"]["data"], columns = data_json["x_var"]["columns"], index = data_json["x_var"]["index"])
    df_json['dh_profile'] = [data_json['dh_profile']]
    df_json = df_json.explode('dh_profile')
    df_json.index = np.arange(0,df_json.shape[0]*3, 3)
    df_json['coil_id'] = '00112233'
    lambda1 = data_json["alpha"]
    pass_nr = data_json["pass_number"]
    dh_tol = calc_dh_tol(pass_nr, df_json['REF_INITIAL_THICKNESS'].iloc[0])

    recsys_obj = ZRMrecsys()
    coil_df = recsys_obj.get_steady_state_features(df_json, pass_nr)

    res1 = recsys_obj.recommend(coil_df, pass_nr, dh_tol, lambda1)
    #print(res1[0])
    res2 = recsys_obj.optimize_1(coil_df, pass_nr, dh_tol, lambda1)
    #print(res2)
    res3 = recsys_obj.optimize_2(coil_df, pass_nr, dh_tol, lambda1)
    #print(res3)
    #res = recsys_obj.optimize(coil_df, pass_nr, dh_tol, lambda1)
    
    final_res = pd.concat([res1[0], res2, res3]).astype(float).round(2)
    print(final_res.values)

