# Dimensionality Reduction and its Importance

*Jordon Xue*


Professor Larrel Pinto gave a lecture about dimensionality reduction on November 10th. This blog will cover the content covered in that lecture and illustrates details about dimensionality reduction with a focus on the linear mapping.
***
## What is dimensionality reduction?
In machine learning, data are often high-dimensional. For example, in tasks like document classification, there can be thousands of words per document. Some of those words can be meaningful to help us classify the type of documents. In document classification, the name of the president usually appears in a political article, so it can be seen as meaningful and contribute to our task. However, there are other repetitive words like "the", "I", and "a" that don't provide any meaningful information but are still deemed as dimensions of documents. Hence, it's important for us to reduce those redundant dimensions/features in performing machine learning tasks.

Then, dimensionality reduction, as the name points out, is an unsupervised machine learning technique that helps transform data from a high-dimensional feature space to a low-dimensional feature space with meaningful features. 

If we remain data high-dimensional in our machine learning task, problems of the curse of dimensionality can occur. 

## Curse of dimensionality
Bellman first introduced the concept of the curse of dimensionality and pointed out that various phenomenon that occurs within high-dimensional space does not exist in low-dimensional settings. Due to the higher number of dimension, the model gets sparse. Higher dimensional space causes a problem in clustering (becomes very difficult to separate one cluster of data from another), search space also increases, and the complexity of the model increases. Below are several ways of the curse of high dimensionality.

1) Redundant or irrelevant features degrade the performance. 

2) Difficult to interpret and visualize

3) Computation becomes infeasible for some algorithms

4) Hard to store high-dimensional data

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202030723-a9c3675c-91e0-46fe-937f-ae9580b04f97.png">
</p>

Figure above demonstrates the amount of training data needed to cover 20% of the feature range grows exponentially with the number of dimensions.

High-dimensional data gives more information for machines to learn and figure out. Hence, the irrelevant data create more noise in the model. The model then may learn from the noise and overfit, so it cannot generalize well. Moreover, when data contains high dimensions, it’s hard and nearly impossible to visualize the data. If data can be compressed into a few dimensions that matter, it’ll be much easier to interpret and visualize the data. Also, high-dimensional data requires more complexity for algorithms to learn and train, so it not only creates extra difficulties to store the data but also makes the computation infeasible for some algorithms like the random forest.

Therefore, we need dimensionality reduction to avoid the curse of dimensionality.

## Options for dimensionality reduction
One option is feature selection. Intuitively, when there're so many features for data, we can ask experts to advise us to select meaningful features and abandon redundant features. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202039132-5b7273ea-718f-4688-8d02-2ec0217971a9.png">
</p>

This process of feature selection is usually done manually and intuitively. It requires domain knowledge from experts and can result in a loss of information due to the selection process.

Another option is model regularization, which selects and tunes the preferred level of complexity to help the model generalize better. For example, in regression, when we want to regularize the model, we adopt loss minimization or optimization (like L1 or L2 norm functions used). In the process, some features are assigned a weight of 0 or close to 0, which basically inform us that those features are not meaningful to help predict the output in our model. 

However, model regularization requires supervised learning such as linear regression because we are making a prediction with features (such that we have a target). Hence, it's not in the domain of unsupervised dimensionality reduction I want to discuss in this blog.
***
## Linear dimensionality reduction
One option to conduct unsupervised dimensionality reduction is by learning a mapping from high-dimensional to low-dimensional space, which can be either linear or non-linear. Linearly, the mapping is in the form of a matrix such that matrix z = A * matrix x to project x to z. In machine learning, an approach we usually use linear dimensionality reduction is principal component analysis (PCA).

## Principal component analysis
PCA is defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202081526-a00b7e2a-3f7e-41cf-8286-f45fb6c0db64.png">
</p>

Intuitively, for linear transformation we want the purple line in the above figure because it represents the directions of the data that explain a maximal amount of variance, that is to say, the lines that capture most information of the data. In the figure above, when the black line matches the purple, it goes through the origin and it’s the direction in which the projection of the points (red dots) is the most spread out. Mathematically speaking, it’s the line that maximizes the variance (the average of the squared distances from the projected points (red dots) to the origin). 

In PCA, the purple line is the first principal component. The second line (component) then will be in the orthogonal direction to it. Then we can find out how much variance each principal component can be actually accounted for. The goal of PCA is to use a few principal components obtained to reduce the dimensions of data, while still explaining a high amount of variance. 

To efficiently perform a PCA task, we first standardize our data. Since PCA is very sensitive to variance, the aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis. if there are large differences between the ranges of initial variables, those variables with larger ranges will dominate over those with small ranges, which will lead to biased results, so we need to scale the data to make it comparable. Mathematically, we can use the z score to normalize the data: z = (value - mean) /standard deviation. 

Then, the second step to perform PCA efficiently is computing the covariance matrix C. Covariance will help us to understand the relationship between the mean and original data. Since variables are sometimes highly correlated in such a way that they contain redundant information, we compute the covariance matrix to identify the correlation/relationships.

The covariance matrix is a p × p symmetric matrix (p is the number of dimensions) that has as entries the covariances associated with all possible pairs of the initial variables. Based on properties of covariance, since Cov(a, a)=Var(a), in the main diagonal are the variances of each initial variable. Also, since Cov(a,b)=Cov(b, a), the entries of the covariance matrix are symmetric to the main diagonal, which means that the upper and the lower triangular portions are equal. An example of a 5 × 5 covariance matrix is as below:

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202088050-10b3f700-231e-4e6a-923e-e7fa140b02b3.png">
</p>

Following that, the next step is to calculate the eigenvectors and eigenvalues. Eigenvectors are a special set of vectors that help us to understand the structure and the property of the data that would be principal components, and eigenvalues help us to determine the principal components. The kth largest eigenvalues and their corresponding eigenvectors make the kth most important principal components. Hence, the largest eigenvalue with its eigenvector corresponds to the first principal component. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202092096-f2b8f93f-bb5c-4c02-916f-f943b9a37567.png">
</p>

In linear algebra's sense, eigenvectors of the covariance matrix C are the directions of the axes where there is the most variance and that we call principal components. Eigenvalues are the coefficients attached to eigenvectors, which give the amount of variance carried in each principal component. As you can see from the visualization of principal components and variance explained (can be calculated as λk/Σi λi) from the figure above, PCA tries to put the maximum possible information in the first component, then maximum remaining information in the second, and so on. In this sense, by ranking the eigenvectors in descending order of their eigenvalues, we can get the principal components in order of significance, which explains why eigenvalues correspond with pricinpal components.

Finally, based on the eigenvectors and eigenvalues, we can decide whether to discard some features based on their significance and kept feature vectors that had all meaningful vectors. Then, we got the final data set from the dot product of the feature vector and the scaled data set derived. If we chose to keep k dimensions for the feature vectors, the final data set will be in k dimensions. Therefore, we achieve our goal of dimensionality reduction!

To perform the task in python, we can use the following code:

```
# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing or loading the dataset
dataset = pd.read_csv('MachineLearning2022.csv')
  
# distributing the dataset into two components X and Y
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values

# Splitting the X and Y into the
# Training set and Testing set
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# Applying PCA function on training 
# and testing set of X component
# Python can conduct the complicated algorithm within few lines
from sklearn.decomposition import PCA
  
pca = PCA(n_components = 2)
  
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
  
explained_variance = pca.explained_variance_ratio_
```
## Final thoughts
This blog mainly introduces linear dimensionality reduction as a method to reduce dimensions and avoid curse of dimensionality. To actually determine how many pricipal components to keep is another question. Generally speaking, 2-3 components are good for data visualization. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/118228743/202099874-fd2ab9fb-d607-40ab-9e3c-6dfe537008f9.png">
</p>

One popular and simple way is using the Kaiser criterion, which states that we simply drop components with eigenvalues less than 1. Greater than 1 eigenvalues suggests that the corresponding components explain more variance than a single variable, given that a variable accounts for a unit of variance, so those components should be kept.

Picture references: 

https://neptune.ai/blog/dimensionality-reduction#:~:text=Advantages%20and%20disadvantages-,What%20is%20dimensionality%20reduction%3F,variables%20are%20also%20called%20features.

https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/

https://medium.com/odscjournal/a-new-method-of-data-mapping-dimensionality-reduction-network-theory-bf9101c6c165

http://www.statpower.net/Content/312/R%20Stuff/PCA.html

https://builtin.com/data-science/step-step-explanation-principal-component-analysis
