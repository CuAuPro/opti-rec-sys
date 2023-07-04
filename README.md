# opti-rec-sys
Recommender system for process optimization.

# Abstract - Master Thesis

Cold rolling of sheet metal is one of the most important processes in sheet metal processing, which is used to reduce the thickness, equalize the thickness and ensure the appropriate mechanical properties of the material. In order to achieve consistently high product quality and maximum productivity of the equipment, it is important to properly adjust and adapt the process parameters to the current process tasks.

The focus of this [thesis](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=147379&lang=slv) is on the development of a recommender system for operator decision support in industrial processes. The main goal of the thesis is to verify if the practice of recommender systems, commonly used in the field of marketing, can be transferred to the field of industrial decision support systems. The goal of such a system is to extract the most effective operator practices, transfer them to less experienced operators, and use them to automatically adapt process settings to current process tasks.

A special tool has been developed to demonstrate the use of the developed decision support system in an industrial process. This tool is intended for operators to adjust recipe parameters according to the characteristics of the individual workpiece, and for technologists to determine more appropriate general recipes for a wider range of products to be manufactured. The developed system can be used to identify alternatives and improvements compared to current practice, taking into account previous knowledge and experience.

The developed operator tool and recommender system approaches are not
limited to the specific characteristics of the considered rolling mill, which allows transferability and application of these approaches also in related production processes with appropriate adaptation.

# Dataset

The dataset contains a collection of process values obtained from a process line. Each value represents the mean measurement within a window at specific intervals along the distance domain. These intervals were equidistant and sampled under steady-state conditions, ensuring consistent data collection.

The process values included in the dataset cover a range of parameters and variables relevant to the process line. These values provide information about the behavior and characteristics of the process at different points along the distance domain.



Dataset can be downloaded from [Zenodo](https://zenodo.org/record/8085933), doi=10.5281/zenodo.8085933, and placed into `data/`.

# System train

Associate code is `RecSys_train.ipynb`.


In this section, we will cover the training process for two recommender systems:
 - one based on memory and 
  - based on a model. Additionally, we will discuss how the NeuMF (Neural Matrix Factorization) model is adapted for process optimization using the NeuMF+ method.

## Dataset Preparation

Before training the recommender systems, it is essential to prepare the dataset. The dataset should contain relevant information about the process line, such as historical data on process values, measured parameters, and any other variables that can contribute to the recommendation process.

## Recommender System Based on Memory

The first recommender system is based on memory. This approach utilizes the historical data to make recommendations by finding similar patterns or instances in the dataset. By identifying similarities between current process conditions and past data, this system can suggest optimal settings or actions for the process line.

To implement the memory-based recommender system, various techniques such as collaborative filtering or content-based filtering can be employed. These techniques leverage the similarity between process conditions or the characteristics of the process line to provide recommendations.


## Recommender System Based on Model (NeuMF+)
The second recommender system utilizes a model-based approach, specifically the NeuMF+, which is modification of Neural Matrix Factorization (NeuMF) model. NeuMF is a deep learning model that combines matrix factorization and neural networks to capture complex patterns and relationships in the data.

For the process optimization task, the NeuMF model is adapted and enhanced using the NeuMF+ method. NeuMF+ incorporates an optimization of recipe parameters technique, which involves adjusting the process parameters and optimizing the recipe to achieve desired outcomes. This approach allows the recommender system to suggest optimized settings for the process line based on the specific goals and requirements.

# System inference

Associate code is `RecSys_inference.ipynb`.

In this section, we will focus on the inference phase of the two recommender systems discussed earlier: the memory-based system and the model-based system (NeuMF+).

During the inference phase, both systems can be utilized to provide recommendations and optimize the process line based on the trained models.

Comparing the two systems, the memory-based recommender system relies on historical data and similarity matching to generate recommendations. It considers patterns and instances from the past to make suggestions for the current process conditions. On the other hand, the model-based system (NeuMF+) utilizes the trained neural network model to capture complex relationships and patterns in the data, enabling it to make more accurate and personalized recommendations.

By using both systems in the inference phase, process operators or engineers can compare the recommendations generated by each system and evaluate their effectiveness. This comparison can provide insights into the strengths and weaknesses of each system and help in decision-making for process optimization.

Ultimately, the goal of the inference phase is to leverage the trained recommender systems to make informed decisions, improve process performance, and optimize recipe parameters based on the specific objectives and constraints of the process line.


# Acknowledgments

The work was carried out within the framework of the international project INEVITABLE ("Optimization and performance improving in metal industry by digital technologies")
(GA No. 869815), co-funded by the European Commission
under the Horizon 2020 program, SPIRE, and the national
research program Systems and Control, P2-0001.

# License

Please refer to the LICENSE file included in the repository for information regarding the usage and distribution of the dataset and source code.