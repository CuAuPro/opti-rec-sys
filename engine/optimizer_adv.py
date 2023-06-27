
import torch
import pandas as pd
import numpy as np
from modelling.preprocessing import get_bin_labels
from engine.utils import most_similar_index
from typing import List
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
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
        self.lambda2 = 1-self.lambda1
        self.dh_tol = dh_tol #um
        self.model = model
        self.pass_nr = pass_nr
        self.sc_recipe = sc_recipe
        self.recipe_dict = recipe_dict
        self.categorizers = categorizers

    class OptSampling(Sampling):
        def __init__(self, initial_guess, var):
            super().__init__()
            self.initial_guess = initial_guess
            self.cov_matrix = np.identity(initial_guess.shape[0])*var

        def _do(self, problem, n_samples, **kwargs):
            X = np.random.multivariate_normal(self.initial_guess, self.cov_matrix, n_samples)
            X = np.clip(X, problem.xl, problem.xu)
            return X


    class OptProblem(Problem):
      def __init__(self, opt, n_var, n_obj=1, n_constr=0, xl=np.array([]), xu=np.array([]), args=None):
          super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

          self.opt = opt
          self.args = args

      def _evaluate(self, x, out, *args, **kwargs):
          coil_idx = self.args[0]
          coil_features = self.args[2]
          coil_idx = np.repeat(coil_idx,x.shape[0],axis=0)
          coil_features = np.repeat(coil_features,x.shape[0],axis=0)

          recipe_columns = ['tension_en', 'tension_ex', 'velocity_mdr']
          x_in = pd.DataFrame(x.reshape(-1,len(recipe_columns)), columns=recipe_columns)
          x_in_bin = x_in.copy()
          for column in recipe_columns:
            x_in_bin[column] =  get_bin_labels(x_in[column].values, self.opt.categorizers[column], padding=2)
          #print(x_in)
          recipe_idx = x_in_bin[recipe_columns].astype(str).apply(lambda x: ''.join(x), axis=1)
          recipe_idx = recipe_idx.values
          #print(recipe_idx)
          del x_in_bin

          exists_flags = [idx in self.opt.recipe_dict.keys() for idx in recipe_idx]
          recipe_idx = [idx if flag == True else most_similar_index(idx, list(self.opt.recipe_dict.keys())) for idx, flag in zip(recipe_idx, exists_flags)]

          recipe_idx = ([self.opt.recipe_dict[pid] for pid in recipe_idx])
          #print(recipe_idx)
          recipe_idx = torch.Tensor(recipe_idx).to(torch.int32)
          #print(x_in)
          x_in = torch.Tensor(self.opt.sc_recipe.transform(x_in)).to(torch.float32)

          #print(coil_features)
          y = self.opt.model(coil_idx, recipe_idx, coil_features, x_in).detach().numpy()

          J1 = y
          #print(J1)
          new_speed = self.opt.sc_recipe.inverse_transform(abs(x_in.detach().numpy()))[:,-1]
          #print(new_speed)
          J2 = (max(self.opt.bounds[self.opt.pass_nr][2]) - new_speed)
          J1 = J1/np.linalg.norm(self.opt.dh_tol) #normalize to tolerance
          J2 = J2/max(self.opt.bounds[self.opt.pass_nr][2])
          #print(self.lambda2*(J2**2))

          J3 = sum(abs(y[y>self.opt.dh_tol] - self.opt.dh_tol))
          #print("-----")
          #print(J1)
          #print("-----")
          #print(J2)
          #print("-----")
          #print([J1,J2,J3])
          Jtot = self.opt.lambda1*(J1**2) + self.opt.lambda2*(J2**2) + (J3**2)

          out["F"] = Jtot

    def optimize(self, x0: np.array, x_args: torch.Tensor, coil_index: torch.Tensor, recipe_index: torch.Tensor, bounds: np.array=None, verbose=False):
        """Optimization wrapper

        Args:
            x0 (np.array): Initial recipe
            x_args (torch.Tensor): Additional static (coil) features
            coil_index (torch.Tensor): Index of coil
            recipe_index (torch.Tensor): Index of recipe
            bounds (np.array, optional): Tight bounds
            verbose (bool): Verbose

        Returns:
            _type_: Optimization result
        """
        if bounds is None:
            bounds = self.bounds[self.pass_nr]

        sampling = self.OptSampling(x0.flatten(), x0.flatten()*0.10)
        problem = self.OptProblem(self, n_var=x0.shape[1], n_obj=1, n_constr=0,
                             xl=bounds[:,0], xu=bounds[:,1],
                             args=[coil_index, recipe_index, x_args, self.sc_recipe])
        
        algorithm = PSO(pop_size=50, sampling=sampling)
        #termination = get_termination("n_gen", 100) #leave default
 
        res = minimize(problem,
               algorithm,
               #termination,
               seed=1,
               verbose=verbose)

        return res