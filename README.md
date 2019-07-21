# Decision-Trees
Most of real world ML problems are generally focused on Regression and Classification problems. In this repo we will discuss decision trees in full depth using a widely used classification problem IRIS Species classification.

<h3> Let's Start</h3>
Dataset details.....
Iris challenge consists of 150 image samples from different species Iris-setosa,Iris-versicolor,Iris-virginica. We need to classify samples into these species.
Get info about dataset and description of challenge - https://www.kaggle.com/uciml/iris

<h4>Let's jump into understanding decision trees-</h4>
A decision tree is a flowchart-like structure in which each internal node represents a test on a feature (e.g. whether a coin flip comes up heads or tails) , each leaf node represents a class label (decision taken after computing all features) and branches represent conjunctions of features that lead to those class labels. The paths from root to leaf represent classification rules.
Decision trees are constructed via an algorithmic approach that identifies ways to split a data set based on different conditions. It is one of the most widely used and practical methods for supervised learning. Decision Trees are a non-parametric supervised learning method used for both classification and regression tasks.

<h3>Terminologies related to Decision Tree</h3>
•	Root node: the topmost tree node which divides into two homogeneous sets.</n>
•	Decision node: a sub-node which further splits into other two sub-nodes.
•	Terminal/Leaf node: the lowermost nodes or the nodes with no children that represents a class label (decision taken after computing all attributes)
•	Splitting: dividing a node into two or more nodes. The splitting technique results in fully grown trees until the criteria of a class attribute are met. But a fully-grown tree is likely to over-fit the data which leads to poor accuracy on unseen observations. This is when Pruning comes into the picture.
•	Pruning: Process of reducing the size of the tree by removing the nodes which play a minimal role in classifying an instance without reducing the predictive accuracy as measured by a cross-validation set.
•	Branch: a sub-section of a decision tree is called a branch

<h3>Assumptions in DT </h3>
•	In the beginning, the whole dataset is considered the root.
•	Best attribute is selected as the root node using some statistical approach.
•	Records are split recursively to produce homogeneous sub-nodes.
•	If the attributes are continuous, they are discretized before building the model.

<h3>Pseudo-code for Decision Tree Algorithm </h3>
1.	Select the most powerful attribute as the root node.
2.	Split the training set into sub-nodes such that each sub-node has identical attribute values.
3.	Repeat Step 1 and 2 until you meet the criteria of the class attribute.
4.	Perform pruning or remove unwanted nodes if you have a fully grown tree such that it doesn’t affect the prediction accuracy.


<h3>The primary differences and similarities between Classification and Regression Trees are:</h3>
1.	Regression trees are used when dependent variable is continuous. Classification Trees are used when dependent variable is categorical.
2.	In case of Regression Tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value.
3.	In case of Classification Tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
4.	Both the trees divide the predictor space (independent variables) into distinct and non-overlapping regions.
5.	Both the trees follow a top-down greedy approach known as recursive binary splitting. We call it as ‘top-down’ because it begins from the top of tree when all the observations are available in a single region and successively splits the predictor space into two new branches down the tree. It is known as ‘greedy’ because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
6.	This splitting process is continued until a user defined stopping criteria is reached. For e.g.: we can tell the algorithm to stop once the number of observations per node becomes less than 50.
7.	In both the cases, the splitting process results in fully grown trees until the stopping criteria is reached. But, the fully grown tree is likely to over fit data, leading to poor accuracy on unseen data. This bring ‘pruning’. Pruning is one of the technique used tackle overfitting.

https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb

There are three commonly used impurity measures used in binary decision trees: Entropy, Gini index, and Classification Error.
Entropy (a way to measure impurity):
Gini index (a criterion to minimize the probability of misclassification):
Classification Error:

The entropy is 0 if all samples of a node belong to the same class, and the entropy is maximal if we have a uniform class distribution. In other words, the entropy of a node (consist of single class) is zero because the probability is 1 and log (1) = 0. Entropy reaches maximum value when all classes in the node have equal probability.
 

Algorithm – explained using entropy		
The core algorithm for building decision trees called ID3 by J. R. Quinlan which employs a top-down, greedy search through the space of possible branches with no backtracking. ID3 uses Entropy and Information Gain to construct a decision tree. 		
		
Entropy		
A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values (homogenous). ID3 algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one.		
 		
 		
How to avoid overfitting when using ID3?
The ID3 algorithm grows each branch of the tree just deeply enough to perfectly classify the training examples. While this is sometimes a reasonable strategy, in fact it can lead to difficulties when there is noise in the data,or when the number of training examples is too small to produce a representative sample of the true target function. In either of these cases, this simple algorithm can produce trees that overfit the training examples.
 
Overfitting in decision tree learning. As ID3 adds new nodes to grow the decision tree, the accuracy of the tree measured over the training examples increases monotonically. However, when measured over a set of test examples independent of the training examples, accuracy first increases, then decreases.
One way this can occur is when the training examples contain random errors or noise. Moreover, overfitting is possible even when the training data are noise-free, especially when small numbers of examples are associated with leaf nodes. In this case, it is quite possible for coincidental regularities to occur, in which some attribute happens to partition the examples very well, despite being unrelated to the actual target function.
