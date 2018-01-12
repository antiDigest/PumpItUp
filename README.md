# PumpItUp
Pump It Up: Data Mining the Water Table


# About

The task of this project is to predict which water pumps throughout Tanzania are functional, which of those need repairs, and which do not work at all. The prediction is based on a number of variables about what kind of pump is operating, when it was installed, and how is it managed. We are comparing the performance of a couple of boosted classifiers with Support Vector Machines (SVMs) and the performance of SVMs with different kernel functions. The selected boosted classifiers for this project will be XGBoost and LightGBM. The ultimate goal would be to find the best classifier and suggest improvements. Another aim of this project was to be able to get familiar with popular machine learning libraries like scikit-learn, xgboost and lightgbm.

# Data Set

There are 40 columns in the training dataset with over 59,000 instances of data. Each of these instances have a classifying label which classifies the data point to either “functional”, “non functional” and “functional needs repair”. The labels mean the following:

*	functional - The waterpoint is operational and there are no repairs needed

*	functional needs repair - The waterpoint is operational, but needs repairs

*	non functional - The waterpoint is not operational

# Feature Engineering

We tried all types of encoders out of which LabelEncoder from Scikit Learn library was selected. Another option was one-hot encoding of all the categorical features, which resulted in more than 59,000 features. This approach was abandoned because the number of features was almost equal to the number of instances, which would had never gotten done with even principal component analysis. Label Encoder on the other hand converts categories to nominals which can be accessed by the classifier as numbers but may not always be the best approach to machine learning classification tasks. We had to take this approach in order to compensate for time consumption.

After encoding of the features, we found that some of the features were almost exact copies of some other features and some features were not changing. These attributes like, num_private, funder, scheme_name, installer etc have not been used in any experiments from this point on.

# Principal Component Analysis

Principal component analysis was run over the attributes to reduce the dimensionality of the data. The final selection was a dataset with 20 attributes which essentially seemed to give the best performance among all tested. 

# Classifiers and Training

Gradient Boosted Trees (XGBoost Library) have been used with a learning rate of 0.03, with maximum allowed depth of a single tree being 18. We generate a maximum of 400 estimators or trees for the ensemble and keep a regularization constant gamma as 3 for the complete training. It was run on 6 cores of an octa core machine.

DART algorithm (LightGBM Library) worked very well on a completely different set of parameters and gave us results similar to Gradient Boosted Trees. In DART, we set the maximum number of leaves generated as 173, and maximum allowed depth of a single tree as 14. The learning rate was set to 0.2 which was about ten times what we used with MART algorithm. The number of estimators was set to a maximum of 350 and just like MART algorithm, it was run over 6 cores of an octa core machine. The objective in both of these algorithms was multiclass classification by maximizing the logistic loss function.

SVM was again running on 6 cores of an octa core machine. The kernel was set to ‘rbf’ and the regularization constant gamma was set to 0.001. We tested with various values of C to see an improvement in performance, but the best performance was with C around 1 or 2.

All of the classifiers were also tested with a 5-fold cross validation. The results section discusses the outputs of all the classifiers compared to the expectations we had with them.

# Results

MART (Gradient Boosted Trees) algorithm or gradient boosted trees was expected to work the best over the given data, but the DART algorithm might have outperformed it with an accuracy of 79.2% as the best performance of all with 20 principal components extracted using the Scikit Learn PCA.

SVM, which was expected to run better than or at least similar to other algorithms gave us some disappointing results. With kernel set to “poly”, SVM took a lot of time to converge given 6 cores on an octa core machine. Hence, kernel “poly” was not used in any classification. The ‘rbf’ kernel was pretty fast in converging as well as gave the best accuracy among all the others. The ‘sigmoid’ kernel runs considerably fast but fails to give an accountable accuracy ( < 50% ).

# Future Work

There are still a lot of things that can be improved in this project. I might not have a lot of time to invest in it again as it was a simple machine learning course project, but I encourage all interested to work on it and ask questions if need be.