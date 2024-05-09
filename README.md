# Predicting Student Graduate Success with Machine Learning

## Abstract

Machine learning has shown great potential in improving education outcomes, and this paper aims to contribute to this growing field by applying machine learning techniques to predict stu- dent graduate success. The study utilizes a public Kaggle dataset and evaluates various machine learning algorithms to determine the most accurate model for predicting graduate success. Results indicate that machine learning algorithms such as SVM, Neural Network, and Logistic Regression can accurately predict graduate success with an accuracy of over 90 percent. These findings have significant implications for educational institutions seeking to improve the percentage of graduating students.

## Data preprocessing

The raw dataset [2][7] is sourced from Kaggle and contains 34 features predicting 3 target variables: ’Enrolled’, ’Dropout’, and ’Graduate’. The dataset includes a total of 4424 instances, with the respec- tive number of instances per target variable being 794 for ‘Enrolled’, 1421 for ‘Dropout’, and 2209 for ‘Graduate’, as shown in Figure 1. Every instance in the dataset is anonymous and it is publicly avail- able, therefore there are no ethical concerns or limitations in its use. The dataset contains no missing
values and features a mix of binary, integer, and float values and target variables are represented as strings.

### Delete negligent target variable

The research question of this paper aims to predict graduate success; therefore, the target variable ‘Enrolled’ was deemed negligent as an enrolled student has neither dropped out nor graduated. As a result, all instances in the dataset with the ’Enrolled’ target variable, totaling 794, were eliminated, reducing the total number of instances to 3630.

### Feature elimination

In order to remove any irrelevant or corrupting features from the dataset, Recursive Feature Elimina- tion [9] was used via the Scikit-learn (sklearn) package. The script ranked the features by significance and returned the bottom five ranking features for deletion, as illustrated in Appendix 1. However, no consistently negligent features were found and therefore all were deemed significant.

### Undersampling

At this point in the preprocessing pipeline, the ratio of instances between ‘Dropout’ and ‘Graduate’ classes were 1421/3630 and 2209/3630, respectively. Given the class imbalance, the majority class (‘Graduate’) was undersampled in order to achieve a balanced distribution of instances, with a 50-50 split of 1421 instances for each class. This step was carried out to prevent potential errors in bias towards the majority class or overfitting the majority class.

### Converting string values

As the machine learning methods in Python require numpy arrays as input, it was not feasible to use string values for the target variables. As a solution, these variables were converted into binary values, with 0 representing ’Dropout’ and 1 representing ’Graduate’.

### Splitting training set and testing set

The final step was splitting the preprocessed dataset into training and test set with an 80:20 split respectively. This test set then stayed unchanged throughout the analysis and comparison of methods in order to ensure consistency in results.


## Methods

### K-Nearest Neighbour

K-Nearest Neighbour (kNN) is a supervised learning classifier [1], which uses proximity to make classifi- cation and predictions. Using the student graduate success dataset, that builds on student demographic and academic measures, kNN predicts a new instance’s values after categorizing it to the k number of nearest neighbors based on their similarity. As kNN is a simple, yet effective algorithm as only the value of k can be changed and optimized for. To do so a brute force script was created, see Appendix 2, and the k value correlated with the highest accuracy was selected which was identified to be k=59.

### Logistic Regression

Logistic regression is a statistical method used to model the relationship between a binary dependent variable and one or more independent variables. It is a type of generalized linear model that uses a logistic function to map the linear combination of the independent variables to a probability of the dependent variable taking a particular value. The logistic function, also known as the sigmoid function, takes any real-valued input and maps it to a value between 0 and 1, representing the probability of the dependent variable being 1. The model estimates the values of the coefficients that maximize the likeli- hood of observing the data given the model. This is achieved by minimizing the negative log-likelihood of the data, which is equivalent to maximizing the log-likelihood. It has several advantages over other classification methods. It is relatively robust to noise and outliers in the data, making it suitable for real-world applications. It can be efficiently trained using optimization algorithms, allowing it to handle large datasets. Additionally, The coefficients of the logistic regression model can be interpreted as the effect of the corresponding independent variables on the log-odds of the dependent variable, providing valuable insights into the relationships between the variables. This was optimized via Grid- SearchCV and the selected optimal values were C=0.001, maxiter=500, penalty=’none’, solver=’sag’. Under these optimal values, the accuracy went from 0.89 to 0.90, whereas precision and recall stayed the same, so there was no significant improvement in the overall performance.

### Naive Bayes

Naive Bayes algorithms are probability based algorithms, which build on Bayes’ Theorem. The method assumes that the different features are independent of each other, and while this is not necessarily true in the real world, these algorithms usually present strong results. Moreover, Naive Bayes can be extremely fast compared to other methods. We have implemented our algorithm in the context of a Gaussian (normal) distribution [3], where the algorithm also assumes that the data follows a normal distribution. The algorithm without prior hyperparameter optimization produces relatively low results, with the accuracy of 0.8242. The only parameter that can be optimized is ‘var-smoothing’, which adds a value to the distributions variance, for which the default starting value is calculated from the training data set. This widens the distribution curve and allows values to be more spread out, even if they are more far away from the distribution mean. The optimal value for ‘var-smoothing’ was found to be 0.0001 with the GridSearchCV algorithm, which increased the accuracy to 0.8312.

### Random Forest

Random Forest is a machine learning algorithm that belongs to the family of ensemble methods, which combine multiple individual models to make a more accurate prediction than any of the individual models alone. Random Forest is based on the concept of decision trees, which recursively partition the feature space into smaller regions to make a classification or regression prediction. However, unlike decision trees that can be prone to overfitting, Random Forest reduces the variance of the model by combining the predictions of multiple decision trees that are trained on randomly selected subsets of the training data.This randomness in the training data and feature selection helps to decorrelate the decision trees and make the model more robust to noise in the data. The performance of Random Forest depends on several hyperparameters, such as the number of decision trees in the ensemble, the maximum depth of each decision tree, and the minimum number of samples required to split an internal node. These hyperparameters can significantly affect the performance of the model, and therefore, it is essential to tune them appropriately to achieve the best performance on the given dataset. These hyperparameters are: n-estimators: the number of trees in the forest, max-features: the number of features to consider when looking for the best split, max-depth: the maximum depth of the tree, min-samples-split: the minimum number of samples required to split an internal node, min- samples-leaf: the minimum number of samples required to be at a leaf node and bootstrap: whether or not to use bootstrapping to sample the data. GridSearchCV found the optimal hyperparameters to be: n-estimators: 90, max-features: ’sqrt’, max-depth: 4, min-samples-split: 2, min-samples-leaf: 1, bootstrap: True. This left the accuracy unchanged at 0.88.

### Support Vector Machines 

Support vector machines (SVMs) are another type of supervised learning algorithm. Through the training data, SVMs are trying to find the optimal hyperplane to separate the classes. The model uses a margin, which is the distance between the hyperplane and the closest data points from each class. SVMs can also be used for non-linear classification, where the model uses a kernel function, which turns the input data into a higher-dimensional space. In our implementation we used a basic linear kernel and a radial basis function (RBF) kernel. There were two hyperparameters that we tried to optimize, namely C and gamma. C defines the error tolerance of the margins (how many misclassifications are tolerable) and gamma, in the context of the RBF, defines how far the influence of a single training example reaches. GridSearchCV found the optimal values to be C=100, gamma=1 and kernel=linear which yielded an accuracy of 0.91.
Figure 3: Visualisation of kernel function.

### Ridge Regression 

Ridge regression is a statistical method used in regression analysis to address the phenomenon where independent variables in a regression model are highly correlated with each other. This correlation leads to unstable and unreliable estimates of the regression coefficients, which can result in incorrect predictions and misleading interpretations. Ridge regression is a regularization technique that adds a penalty term to the sum of squared errors. The penalty term is proportional to the square of the magnitude of the regression coefficients, and it is controlled by a hyperparameter called the regular- ization parameter or shrinkage parameter, denoted by alpha. The intuition behind ridge regression is that the penalty term shrinks the regression coefficients towards zero, reducing their variance and making them more stable and reliable. However, the penalty also introduces bias into the estimates, which can lead to underfitting if the regularization parameter is too large. GridSearchCV found the optimal values to be alpha=1 which yielded an accuracy of 0.90.

### Perceptron

A perceptron is a type of neural network, inspired by the processes of the human brain. It is a binary classifier that classifies data into two categories. It works by taking in an input vector and assigning weights to each of its elements. The input vector is then multiplied by the weight vector, and the result is passed through an activation function (ReLU, sigmoid, softMax etc.). The output of the activation function is the predicted class label of the input vector. During the training phase, the algorithm adjusts the weights of the input vector based on the error between the predicted class label and the actual class label. It keeps iterating until the weights are adjusted such that the error is minimized. By using the GridSearchCV function, the following optimal parameters were obtained : learning rate - 0.0001, tolerance - 1e-05. In this specific scenario, the accuracy of the perceptron algorithm went from 0.89 to 0.85. There are a few possible explanations to why this happened, such as overfitting during the initial training phase that can happen if the model is too complex or if the training data is not representative of the population. However, the decrease can be also a result of data variability or randomness. In order to improve the performance significantly, the perceptron can be transformed into a deep neural network (as explained in the next section). DNNs main advantage over perceptrons is their ability to learn more complex and abstract representations of the data. The multiple layers of interconnected neurons allow the DNN to capture more subtle patterns and relationships of the features. This makes DNNs particularly well-suited for handling high-dimensional datasets.

### Decision Tree

Decision trees are a type of supervised learning algorithm that recursively partitions the feature space into smaller regions, with the goal of making accurate predictions for unseen data points. The algorithm works by selecting the best feature to split the data into subsets that are as homogeneous as possible with respect to the target variable. The homogeneity of the subsets is measured using a metric such as information gain, gain ratio, or Gini impurity. Information gain measures the reduction in entropy of the target variable after the split, while gain ratio normalizes the information gain by the in- trinsic information of the feature. The splitting process continues until some stopping criterion is met. One of the advantages of decision trees is their interpretability. The structure of the tree can be easily visualized and understood, which makes it easier to interpret and explain the predictions made by the model. However, decision trees can be prone to overfitting, especially when the depth of the tree is too high, and the model becomes too complex. This problem can be addressed using pruning techniques such as reduced-error pruning, cost-complexity pruning, or minimum description length pruning. Sev- eral variants of decision trees have been developed to address some of their limitations. For example, Random Forest combines multiple decision trees trained on different subsets of the data and features, with the goal of reducing the variance of the model and improving its accuracy. Gradient Boosted Trees trains decision trees sequentially, with each tree trained on the residual error of the previous tree, with the goal of reducing the bias of the model and improving its accuracy. As you can see in the results, gradient boosting and random forest significantly increased accuracy. The hyperparameters of the decision tree were optimized using GridSearchCV which yielded optimal values of ccp-alpha=0.0, criterion=’gini’, max-depth=9, max-features=’sqrt’, min-samples-leaf=1, min-samples-split=2 and an accuracy of 0.82.

### Gradient Boosting
 
Gradient Boosting is a decision tree based method, which builds on the previous tree by fitting residual errors to improve the next prediction. The algorithm starts out with a simple model, also known as a weak learner. The weak learner’s residuals are calculated, and a new weak learner model is generated based on the residual and then added to the first learner. This repetition is making the model an ensemble one, and results in an improved performance of the model over time. Additionally, the possibility to fine tune Gradient Boosting with parameters is what makes it a powerful algorithm [4]. After using Grid Search Cross Validation, the optimal parameters were obtained: learning-rate of 0.1, max-depth of 4, max-features set to ’sqrt’, min-samples-leaf of 2, min-samples-split of 2 and n-estimators set to 70. This yielded an accuracy score of 0.90.

### Boostrap Aggregation

Bootstrap Aggregation, also known as Bagging, which trains on a dataset that has the same size as the original data set and is sampled with replacement from the original data set [6]. This means that samples are randomly pulled from the original dataset, and “put back”, so that it could be chosen again. As a result, some samples are drawn multiple times, while others are not drawn at all. Bagging is also an ensemble method, meaning multiple runs of independently trained models are run to improve the performance metrics. What makes bagging a robust model, is that multiple models are trained on different subsets of the original dataset. After running Grid Search Cross Validation, the test for accuracy resulted in 0.8681 under the following learning parameters: bootstrap set as True, max-features set to 3, max-samples set to 0.5 and n-estimators set to 100.

### Wide(r) and Deep(r) Neural Network

Artificial Neural Networks are a subset of machine learning methods. They are the successors of perceptrons. Their name and methodology comes from the view of a human brain as a network of interconnected neural pathways that process information. The high level overview of any neural network is that it consists of an input layer, one or more hidden layers, and lastly an output layer. A distinction can be made about NNs based on the ratio of hidden layers to nodes in a single layer. Let A,B be NNs. One says that A is wider NN than B if A has more nodes per layer than B. Similarly one says that B is a deeper NN than A if it contains a larger number of layers. Neural networks are particularly well suited to handle tasks which include complex patterns and nonlinear relationships between variables.
We implemented two types of networks. A relatively wide neural network which consists of one hidden layer, with 102 neurons as well as a comparatively deeper neural network with 4 hidden layers. Since the problem we are trying to solve is not linearly separable the literature we read suggested using at least one hidden layer. However, by Cybenko’s Universal Approximation Theorem we hypothesized that the problem could be solved well by just one layer with many nodes. This hypothesis was rectified when the classification task was completed with relatively high accuracy. When determining the number of nodes, we relied on manual testing as well as the general ‘rule of thumbs’ concerning wide neural networks. After some testing we settled on 3x the number of inputs for the wider network. For the deeper network the main question is how many layers to include. The methodology we used to get our answer is outlined perfectly by the quote from Yoshua Bengio: “Very simple. Just keep adding layers until the test error does not improve anymore “. We ended up choosing 3 hidden layers each containing the same number of nodes as the count of inputs.
For both networks we preferred the rectified linear activation function in the input and hidden layers. We used the sigmoid function for the output function for the deeper neural net. The sigmoid function maps the output value between 0 and 1. Essentially it gives a probability for which class does the current instance in question belongs to. The threshold value for the activation function 0.5, which finely separates the classes. This provides a simple, interpretable output.



## Results

Machine Learning Methods with Corresponding Accuracy on Test Dataset

| Method               | Accuracy |
|----------------------|----------|
| Bagging              | 0.855888 |
| Decision Tree        | 0.822496 |
| Gradient Boosting    | 0.892794 |
| KNN                  | 0.892794 |
| Logistic Regression  | 0.903339 |
| Naive Bayes          | 0.831283 |
| Perceptron           | 0.859402 |
| Random Forest        | 0.876977 |
| Ridge                | 0.899824 |
| SVM                  | 0.910369 |
| NN-Wide              | 0.905097 |
| NN-Deep              | 0.898060 |

T
