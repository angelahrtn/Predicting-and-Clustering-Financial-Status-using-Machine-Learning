# Predicting-and-Clustering-Financial-Status-using-Machine-Learning
# Project Overview
This project was completed as part of a Machine Learning quiz in the third semester. The primary goal was to predict the financial status of individuals based on various economic and demographic features using machine learning models. The dataset contains financial and personal information about individuals, which is used to classify their financial status. Several machine learning models were implemented, tuned, and evaluated to determine which model performed the best in predicting whether an individual is in a good or bad financial state.

The project was carried out using Python, with a focus on classification techniques, hyperparameter tuning, and model evaluation to ensure the best accuracy and robustness.

# Case Description
The central problem addressed in this project is classifying the financial status of individuals based on a variety of features. The dataset includes variables such as:
  - Income Level
  - Age
  - Employment Type
  - Credit Score
  - Loan Information
Financial stability is a critical issue that impacts loan approvals, credit worthiness, and overall financial planning. The goal of this project is to develop machine learning models that accurately classify individuals into different financial statuses (e.g., good or bad financial health) based on the available data.

# Objectives
- Build predictive models to classify individuals into different financial status categories based on economic and personal features.
- Compare machine learning models to identify which algorithm delivers the highest accuracy and performance.
- Tune models to improve predictive accuracy and ensure they generalize well to unseen data.
- Evaluate model performance using key metrics such as accuracy, precision, recall, and F1-score.
- Provide insights into key features that influence financial status classification.

# Project Steps and Features
1. Data Collection and Preprocessing
- The dataset was loaded and preprocessed for analysis. Key steps included:
  - Handling missing values by either imputing them or removing incomplete rows.
  - Encoding categorical variables using Label Encoding and One-Hot Encoding to convert non-numeric data into a format suitable for machine learning models.
  - Scaling numerical features with StandardScaler to ensure that all features contributed equally to model training and clustering.

2. Clustering Model Construction
- Clustering algorithms were built and tested using:
  - K-Means: A popular clustering algorithm that partitions the dataset into k clusters by minimizing within-cluster variance.
  - Inertia (for K-Means): The sum of squared distances of samples to their closest cluster center.

3. Model Evaluation
- Since clustering is an unsupervised learning task, evaluation techniques differ from classification models. The models were evaluated using the following metrics:
  - Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters. Higher silhouette scores indicate better-defined clusters.
 
4. Model Tuning
- For K-Means, the optimal number of clusters was determined using:
  - Elbow Method: A plot of the inertia values to find the point at which adding more clusters does not significantly reduce the inertia.
  - Silhouette Analysis: Evaluating different numbers of clusters to find the one that maximizes the silhouette score.
 
5. Results Interpretation
- K-Means provided clear clusters, with the optimal number of clusters determined using the Elbow Method. The clusters identified different financial profiles based on income levels, credit scores, and other factors.


# Tools
- Programming Language: Python
- Libraries:
  - Pandas: For data manipulation and cleaning.
  - NumPy: For numerical operations.
  - Scikit-learn: For implementing machine learning models, tuning, and evaluating their performance.
  - Matplotlib & Seaborn: For visualizing the data and evaluation metrics (e.g., confusion matrix, feature importance).

# Challenges
- Determining the Optimal Number of Clusters: One of the primary challenges was selecting the optimal number of clusters (k). Using the Elbow Method to analyze inertia helped, but it still required careful interpretation of the results to avoid overfitting or underfitting the clusters.
- Interpreting the Clusters: After applying K-Means, understanding what each cluster represents and how it relates to different financial profiles was a challenge. It required a deep dive into the feature space to label the clusters meaningfully.
- Scalability: K-Means can be computationally expensive for large datasets, and ensuring that the algorithm converged efficiently was time-consuming for fine-tuning the hyperparameters, such as the number of clusters.

# Conclusion
This project successfully applied the K-Means clustering algorithm to group individuals into distinct financial profiles based on demographic and financial data. The Elbow Method helped identify the optimal number of clusters, allowing us to separate the data into meaningful segments. These clusters provide actionable insights into the different financial behaviors of individuals, such as clusters representing high-income, good credit score individuals, and clusters of lower-income individuals with less favorable financial characteristics.

While the project demonstrated the effectiveness of K-Means in grouping financial data, further improvements could involve experimenting with other clustering algorithms (e.g., Hierarchical Clustering or DBSCAN) to enhance cluster identification, as well as exploring more detailed feature engineering to capture more subtle financial behaviors.
