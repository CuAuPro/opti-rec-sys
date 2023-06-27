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
from scipy import ndimage


import engine.settings as settings
from modelling.preprocessing import feature_engineering, get_bin_labels
from engine.optimizer import Optimizer
import engine.optimizer_adv as optimizer_adv
from engine.utils import most_similar_index, importConfig, calculateCriterion, preprocess_profile, calc_dh_tol

##################################################
#               PARAMETERS
simi_thresh_similar_coils = 0.7 # Used for selection of simi_thresh_similar_coils recipes
N_min_similar_coil_classes = 5  # Used for selection of N_similar_coils (meaning binarized coil classes)
                                # recipes if there are less than
                                # N_similar_coils, when simi_thresh_similar_coils used
                     
N_best_recipes = 5          # Used for selection of best performed N_best_recipes recipes
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

class ZRMrecsys():

    # data to be passed into model
    feature_names = ['dh_entry', 'dh_entry_min', 'dh_entry_max', 'dh_entry_std',
                    'h_entry_ref','h_exit_ref','REF_INITIAL_THICKNESS', 'dh_profile']
    
    coil_columns = [ 'h_entry',
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
                      'velocity_mdr']
    quality_column = 'EXIT_THICK_DEVIATION_ABS_AVG'

    
    
    def __init__(self, infer_type=None):
        self.bounds = settings.config["recipe_bounds"]
        if infer_type == None:            
            self._infer_type = 'optimise'
            
        elif any(infer_type == a for a in ['optimise', 'optimise_adv', 'recommend']):
            self._infer_type = infer_type
        
        return

    def optimize(self, coil_df, pass_nr, dh_tol=dh_tol_def, opt_lambda1=opt_lambda1_def, opt_method=opt_method_def):
        # Model based recipe optimization

        final_res, new_coil, new_coil_idx, new_recipe_idx = self.recommend(coil_df, int(pass_nr), dh_tol, opt_lambda1)

        # prepare optimizer
        opt = Optimizer(model, int(pass_nr), dh_tol, sc_recipe, recipe_dict, categorizers, self.bounds, lambda1=opt_lambda1,
                        method=opt_method)

        Nsegments = final_res.shape[0]
        final_res_opt = pd.DataFrame(columns=self.recipe_columns)
        for k in range(Nsegments):

          x0 = final_res[self.recipe_columns].values[k].reshape(1,-1)

          # prepare new_recipe_idx
          x_args = torch.Tensor(sc_coil.transform(new_coil[self.coil_columns].iloc[[k]])).to(torch.float32)
          new_coil_idx_k = torch.Tensor([new_coil_idx[k]]).to(torch.int32)
          new_recipe_idx_k = torch.Tensor([new_recipe_idx[k]]).to(torch.int32)

          # calculate moroe tight bounds (based on recipe_idx bins)
          bounds = self.tight_bounds(new_recipe_idx[k], int(pass_nr))

          # run optimization
          res = opt.optimize(x0, x_args, new_coil_idx_k, new_recipe_idx_k, bounds)
          res_opt = pd.DataFrame(data=res.x.reshape(1, -1), columns=self.recipe_columns)
          
          final_res_opt = pd.concat([final_res_opt, res_opt])

        final_res_opt.index = np.concatenate([[0],new_coil['slice_idx'].values[:-1]])*3

        final_res_opt['pred'] = self.predictQuality(final_res_opt, new_coil[self.coil_columns], new_coil_idx)

        return final_res_opt

    def optimize_adv(self, coil_df, pass_nr, dh_tol=dh_tol_def, opt_lambda1=opt_lambda1_def, opt_method=opt_method_def):
        # Model based recipe optimization

        final_res, new_coil, new_coil_idx, new_recipe_idx = self.recommend(coil_df, int(pass_nr), dh_tol, opt_lambda1)

        # prepare optimizer
        opt = optimizer_adv.Optimizer(model, int(pass_nr), dh_tol, sc_recipe, recipe_dict, categorizers, self.bounds, lambda1=opt_lambda1,
                        method=opt_method)
        Nsegments = final_res.shape[0]

        final_res_opt = pd.DataFrame(columns=self.recipe_columns)
        for k in range(Nsegments):
              
          x0 = final_res[self.recipe_columns].values[k].reshape(1,-1)

          # prepare new_recipe_idx
          x_args = torch.Tensor(sc_coil.transform(new_coil[self.coil_columns].iloc[[k]])).to(torch.float32)
          new_coil_idx_k = torch.Tensor([new_coil_idx[k]]).to(torch.int32)
          new_recipe_idx_k = torch.Tensor([new_recipe_idx[k]]).to(torch.int32)


          # calculate moroe tight bounds (based on recipe_idx bins)
          bounds = self.tight_bounds(new_recipe_idx[k], int(pass_nr))

          # run optimization
          res = opt.optimize(x0, x_args, new_coil_idx_k, new_recipe_idx_k)
          res_opt = pd.DataFrame(data=res.X.reshape(1, -1), columns=self.recipe_columns)
          
          final_res_opt = pd.concat([final_res_opt, res_opt])

        final_res_opt.index = np.concatenate([[0],new_coil['slice_idx'].values[:-1]])*3

        final_res_opt['pred'] = self.predictQuality(final_res_opt, new_coil[self.coil_columns], new_coil_idx)

        return final_res_opt

    def recommend(self, coil_df, pass_nr, dh_tol=dh_tol_def, opt_lambda1=opt_lambda1_def):
        
        new_coil = feature_engineering(coil_df)
        #print('coil_df')
        #print(coil_df)
        #print('new_coil')
        #print(new_coil)
        new_coil = new_coil[self.coil_columns + ['slice_idx']]
        
        # Binarize data
        new_coil_bin = new_coil.copy()
        for column in self.coil_columns:
            new_coil_bin[column] = get_bin_labels(new_coil_bin[column].values, categorizers[column], padding=2)

        coil_idx = new_coil_bin[self.coil_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
        coil_idx = coil_idx.values

        exists_flags = [idx in coil_dict.keys() for idx in coil_idx]
        coil_idx = [idx if flag == True else most_similar_index(idx, list(coil_dict.keys())) for idx, flag in zip(coil_idx, exists_flags)]

        new_coil.loc[:,'coil_idx'] = ([coil_dict[pid] for pid in coil_idx])
        new_coil_idxs = new_coil['coil_idx'].values

        # Filter recipes (minimize fluctuations between recipes)
        #new_coil_idxs = ndimage.median_filter(new_coil_idxs, size=3)

        #print('new_coil_idx')
        #print(new_coil_idx)
        
        # Select most similar coils
        #simi_idxs = np.argsort(simi.flatten())[::-1]
        #simi_idxs = simi_idxs[0:N_similar_coils]
        Nsegments = new_coil_idxs.shape[0]
        final_res = pd.DataFrame(columns=self.recipe_columns)
        new_recipe_idxs_ret = np.array([])
        new_coil_idxs_ret = np.array([])
        for k in range(Nsegments):
          coil_idx = new_coil_idxs[k]

          user_similarities = cosine_similarity(new_coil_bin[self.coil_columns], coil_bins)[0]
          most_similar_idx = np.argsort(user_similarities)[::-1]

          most_similar_idx = most_similar_idx[user_similarities[most_similar_idx] > simi_thresh_similar_coils][:N_min_similar_coil_classes]
          if most_similar_idx.shape[0] < N_min_similar_coil_classes:
            most_similar_idx = np.argsort(user_similarities)[::-1]
            most_similar_idx = most_similar_idx[:N_min_similar_coil_classes]
          
          res = df.loc[df['coil_idx'].isin(most_similar_idx)]
          # Extract only those values where quality_column is not 0 (none of coil used that recipe)
          res = res.loc[res[self.quality_column] != 0]

          res["Jtot"] = calculateCriterion(res["velocity_mdr"].values, res[self.quality_column].values,
                                                self.bounds[pass_nr], dh_tol=dh_tol, lambda1=opt_lambda1)
          idxs_best = np.argsort(res["Jtot"].values)

          res = res.iloc[idxs_best].iloc[0:N_best_recipes]
          new_coil_idx, new_recipe_idx = res[['coil_idx', 'recipe_idx']].median().values
          new_coil_idx = new_coil_idx.flatten()
          new_recipe_idx = new_recipe_idx.flatten()
          new_coil_idxs_ret = np.append(new_coil_idxs_ret, new_coil_idx)
          new_recipe_idxs_ret = np.append(new_recipe_idxs_ret, new_recipe_idx)
          res = pd.DataFrame(res[self.recipe_columns].median()).T
          final_res = pd.concat([final_res, res])

        final_res.index = np.concatenate([[0],new_coil['slice_idx'].values[:-1]])*3
        final_res['pred'] = self.predictQuality(final_res, new_coil, new_recipe_idxs_ret)

        return final_res, new_coil, new_coil_idxs_ret, new_recipe_idxs_ret
    
    def get_recipe(self, coil_df, pass_nr, dh_tol, opt_lambda1=opt_lambda1_def):
        coil_df.loc[:,'pass_nr'] = pass_nr
        coil_df.loc[:,'coil_id'] = '00112233'
        
        if len(coil_df) > 1: 
            coil_df = preprocess_profile(coil_df)
        else: 
            # no profile provided as input 
            import crecsys
            model = crecsys.ZRMrecsys(self._infer_type)
            return model.get_recipe(coil_df, pass_nr, dh_tol, opt_lambda1)

        if self._infer_type == 'optimise' or self._infer_type=='optimize':
            opt_res = self.optimize(coil_df, pass_nr, dh_tol, opt_lambda1)

        elif self._infer_type == 'optimise_adv' or self._infer_type == 'optimize_adv':
            opt_res = self.optimize_adv(coil_df, pass_nr, dh_tol, opt_lambda1)
            
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

        exists_flags = [idx in recipe_dict.keys() for idx in recipe_idx]
        recipe_idx = [idx if flag == True else most_similar_index(idx, list(recipe_dict.keys())) for idx, flag in zip(recipe_idx, exists_flags)]

        recipe_idx = ([recipe_dict[pid] for pid in recipe_idx])
        recipe_idx = torch.Tensor(recipe_idx).to(torch.int32)
        x_suggested = torch.Tensor(sc_recipe.transform(x_suggested)).to(torch.float32)

        y = model(coil_idx, recipe_idx, x_args, x_suggested).detach().numpy()

        return y

      
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
        if upper_limit > self.bounds[pass_nr][i][1]:
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


    df_sections = preprocess_profile(df_json)
    df_sections[['AL', 'CU']] = df_json[['AL', 'CU']].iloc[0]

    recsys_obj = ZRMrecsys()

    res1 = recsys_obj.recommend(df_sections, pass_nr, dh_tol, lambda1)
    print(res1[0].values)
    res2 = recsys_obj.optimize(df_sections, pass_nr, dh_tol, lambda1)
    print(res2.values)
    res3 = recsys_obj.optimize_adv(df_sections, pass_nr, dh_tol, lambda1)
    print(res3.values)

    #final_res = pd.concat([res1[0], res2, res3]).astype(float).round(2)

