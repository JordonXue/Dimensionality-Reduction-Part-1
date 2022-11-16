# Dimensionality Reduction

*Jordon Xue*


Professor Larrel Pinto gave a lecture about dimensionality reduction on November 20th. This blog will cover the content covered in that lecture and illustrates details about dimensionality reduction.
***
## What is dimensionality reduction?
In machine learning, data are oftern high-dimensional. For example, in tasks like document classification, there can be thoundsans of words per document. Some of those words can be meaningful to help us classifcy the type of the documents. In document classification, the name of president usually appears in political article, so it can be seen as meaningful and contribute to our task. However, there are other repeitive words like "the", "I", "a" that don't provide any meaningful information but are still deamed as dimensions of documents. Hence, it's important for us to reduce those redudant dimensions/features in performaing machine learning tasks.

Then, dimensionality reduction, as the names points out, is a unsupervised machine learning technique that helps transform data from high-dimensional feature space to a low-dimensional feature space with more meaningful features. 

If we remain data high-dimensional in our machine learning task, problems of curse of dimensionality can occur. 

## Curse of dimensionality
Bellman first introduced the concept of the curse of dimensionality and pointed out that various phenomenon that occurs within high-dimensional space does not exist in low-dimensional settings. Due to higher number of dimension model gets sparse. Higher dimensional space causes problem in clustering (becomes very difficult to separate one cluster data from another), search space also increases, complexity of model increases. Below are several ways of the curse of high dimensionality.

1) Redundant or irrelevant features degrade the performance. 

2) Difficult to interpret and visualize

3) Computation becomes infeasible for some algorithms

4) Hard to store high dimensional data

![image](https://user-images.githubusercontent.com/118228743/202030723-a9c3675c-91e0-46fe-937f-ae9580b04f97.png)

Figure1↑ The amount of training data needed to cover 20% of the feature range grows exponentially with the number of dimensions

High-dimensional data gives more information for machines to learn and figure out. Hence, the irrelevant data create more noise in the model. The model then may learn from the noise and overfit, so it cannot generalize well. Moreover, when data contains high dimensions, it’s hard and nearly impossible to visualize the data. If data can be compressed to few dimensions that matters, it’ll be much easier to interpret and visualize the data. Also, high-dimensional data requires more complexity for algorithms to learn and train, so it not only creates extra difficulties to store the data but also makes the computation infeasible for some algorithms like random forest.

Therefore, we need dimensionality reduction to avoid curse of dimensionality.

## Options for dimensionality reduction
One option is feature selection. Intuitively, when there're so many features for a data, we can ask experts to advise us to select meaningful features and abandon redundant features. 

![image](https://user-images.githubusercontent.com/118228743/202039132-5b7273ea-718f-4688-8d02-2ec0217971a9.png)

This process of feature selection is usually manually and intuitively. It requires domain knowledge from experts and can result in loss of information due to the selection process.

Another option is model regularization, which selects and tunes the preferred level of complexity to help model generalize better. For example, in regression, when we want to regularize the model, we adopt loss minimization or optimization (like L1 or L2 norm functions used). In the process, some features are assigned a weiight of 0 or closed to 0, which basically inform us that those features are not meaningful to help predict the output in our model. 

However, model regularizatio requires supervised learning such as linear regression because we are making prediction with features (such that we have a target). Hence, it's not in the domain of unsupervised dimensionality reduction I want to discuss in this blog.
***
## Linear dimensionality reduction
One option to conduct unsupervised dimensionality reduction is by learning mapping from high-dimensional to low-dimensional space, which can be either linear or non-linear. Linearly, the mapping is in the form of matrix such that matrix z = A * matrix x to project x to z. In machine learning, an approach we usually use linear dimensionality reduction is principal component analysis (PCA).

## Principal component analysis
PCA is defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
<p align="center"> 
![image](https://user-images.githubusercontent.com/118228743/202081526-a00b7e2a-3f7e-41cf-8286-f45fb6c0db64.png)  
</p>


Intuitevely, for linear transforamtion 
