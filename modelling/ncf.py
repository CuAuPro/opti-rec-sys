import torch
from torch import nn
from .layers import MultiLayerPerceptron

class ZRM_Reccomender(nn.Module):
    def __init__(self,
                  nr_coils: int,
                  nr_recipes: int,
                  nr_coil_features: int,
                  nr_recipe_features: int,
                  n_factors: int=10):
        """ZRM Reccomender System model

        Args:
            nr_coils (int): Number of different coils
            nr_recipes (int): Number of different recipes
            nr_coil_features (int): Number of coil features
            nr_recipe_features (int): Number of recipe features
            n_factors (int, optional): Embedding size. Defaults to 10.
        """
        super(ZRM_Reccomender, self).__init__()
        self.coil_embeddings = nn.Embedding(nr_coils, n_factors)
        self.recipe_embeddings = nn.Embedding(nr_recipes, n_factors)
        self.coil_bias = nn.Embedding(nr_coils, 1)
        self.recipe_bias = nn.Embedding(nr_recipes, 1)

        self.mlp = MultiLayerPerceptron(nr_coil_features + nr_recipe_features, hidden_layers_dim=[16,8], dropout=0.2, output_layer=False)
        self.mlp_ensemble = MultiLayerPerceptron(self.mlp.hidden_layers_dim[-1]+n_factors, hidden_layers_dim=[8,4], dropout=0.2)
        self.n_factors = n_factors

    def forward(self, coil_indices, recipe_indices, coil_features, recipe_features):
        """Forward method

        Args:
            coil_indices (_type_): coil_indices
            recipe_indices (_type_): recipe_indices
            coil_features (_type_): coil_features
            recipe_features (_type_): recipe_features

        Returns:
            _type_: prediction
        """
        coil_embed = self.coil_embeddings(coil_indices)
        recipe_embed = self.recipe_embeddings(recipe_indices)
        coil_bias = self.coil_bias(coil_indices)
        recipe_bias = self.recipe_bias(recipe_indices)
        
        # concatenate coil and recipe features
        features = torch.cat([coil_features, recipe_features], dim=1)
        
        # Matrix Factorization part
        fm_terms = torch.mul(recipe_embed, coil_embed)
        
        # MLP part
        mlp_terms = self.mlp(features)

        # Ensemble part
        ensemble_terms = self.mlp_ensemble(torch.cat([mlp_terms, fm_terms], dim=1))

        #predictions = coil_bias.squeeze() + recipe_bias.squeeze() + mlp_terms.squeeze() + fm_terms.squeeze()
        predictions = coil_bias.squeeze() + recipe_bias.squeeze() + ensemble_terms.squeeze()
        return predictions











class ZRM_ReccomenderOld(nn.Module):
    def __init__(self, nr_coils, nr_recipes, nr_coil_features, nr_recipe_features, n_factors=10):
        super(ZRM_Reccomender, self).__init__()
        self.coil_embeddings = nn.Embedding(nr_coils, n_factors)
        self.recipe_embeddings = nn.Embedding(nr_recipes, n_factors)
        self.coil_bias = nn.Embedding(nr_coils, 1)
        self.recipe_bias = nn.Embedding(nr_recipes, 1)


        self.linear = nn.Linear(nr_coil_features + nr_recipe_features, 1)

        # Added for additional info net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(max(nr_coil_features, nr_recipe_features), 1)
        
        
        self.n_factors = n_factors

    def forward(self, coil_indices, recipe_indices, coil_features, recipe_features):
        coil_embed = self.coil_embeddings(coil_indices)
        recipe_embed = self.recipe_embeddings(recipe_indices)
        coil_bias = self.coil_bias(coil_indices)
        recipe_bias = self.recipe_bias(recipe_indices)
        
        # concatenate coil and recipe features
        features = torch.cat([coil_features, recipe_features], dim=1)
        
        # compute interactions between all pairs of features
        #pairwise_interactions = torch.matmul(recipe_embed, coil_embed.transpose(0, 1))
        #pairwise_interactions = pairwise_interactions.mean(0)
        pairwise_interactions = torch.matmul(recipe_embed, coil_embed.transpose(0, 1)).transpose(0, 1) * features.unsqueeze(-1).unsqueeze(-1)
        pairwise_interactions =  pairwise_interactions.mean(dim=(1,2,3))

        # compute predictions
        ex_terms = self.linear(features)
        fm_terms = pairwise_interactions.squeeze()
        
        predictions = coil_bias.squeeze() + recipe_bias.squeeze() + ex_terms.squeeze() + fm_terms.squeeze()
        
        return predictions