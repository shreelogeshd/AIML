**Car Insurance Claim Prediction**

In this project, I worked on predicting whether a customer will file a car insurance claim using machine learning. The main idea was to understand customer and vehicle details and use them to identify potential claim risks. This type of prediction can help insurance companies make better decisions and reduce losses.

I started by loading the dataset and understanding the features and the target variable. I performed **Exploratory Data Analysis (EDA)** using basic statistics and visualizations to analyze data distributions, identify missing values, observe class imbalance, and detect any unusual patterns.

After gaining insights from EDA, I cleaned and prepared the data for modeling. This included handling missing values, encoding categorical variables, scaling numerical features, and splitting the dataset into training and testing sets. I used pipelines to keep the preprocessing steps structured and reusable.

To make the insights more understandable and visually clear, I also created an interactive **Power BI dashboard**. The dashboard provides better data visualization by highlighting claim trends, customer behavior, and important features, making it easier for non-technical stakeholders to interpret the data and insights.

After that, I trained baseline models such as Logistic Regression and Decision Tree, and evaluated them using **Accuracy, Precision, Recall, and F1-score**. I then implemented advanced machine learning models like **Random Forest, XGBoost**, and **LightGBM** to improve the prediction performance.

I further optimized the models using hyperparameter tuning techniques such as **GridSearchCV and RandomizedSearchCV**. Through feature importance analysis, I identified the key factors that strongly influence insurance claim predictions.

Finally, I compared all models, selected the best-performing model, and saved it using **joblib**. I documented the complete workflow and ensured the project is reproducible. This project helped me gain hands-on experience with the full data science and machine learning lifecycle.
