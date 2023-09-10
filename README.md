# Loan Default Prediction Final Project in Data Science for Business Applications Course
We were given data from the Home Mortgage Disclosure Act (HMDA) and asked to complete both a classification and regression task. The binary classification task was to predict whether a customer will default on a loan or not, and the regression task was to predict the total loan amount for customers accepted to receive a loan. The dataset contained over 100,000 rows with roughly 100 features ranging from the purpose of the loan to an individual's debt to income ratio to property value. 

# Classification Task Results
Our preferred model is Random Forest (ranger) because it creates decision trees using bootstrapped training samples from the clean dataset, like bagging. However, each split in a decision tree is picked from a randomly selected subset of predictors instead of the full set of predictors. This process decorrelates the trees, minimizing gini impurity without introducing significant bias, which improves accuracy. The most important variable in predicting whether or not the loan was denied is the applicant’s total monthly debt to total monthly income (debt_to_income_ratio). The second most predictive variable is whether the obligation arising from the loan would have been initially payable to the financial institution with the Not Applicable value (initially_payable_to_institution3). The third most important variable is the automated underwriting system used to evaluate applications with the Not Applicable value (aus.1). Finally, the fourth most important variable is the applicant’s gross annual income (income). Our preferred model of Random Forest (ranger) accurately classified 166 data points and inaccurately classified 10 of them, resulting in a prediction accuracy of approximately 94.32% with the clean testing dataset.

# Regression Task Results
Property value was the best predictor of the actual approved loan amount. Further, we can see that loan-to-value ratio is a critical predictor of loan amount. These two variables coupled together make a lot of sense, since property_value and loan_to_value_ratio can be used to approximate the loan amount that an applicant is applying for. If we naively run a regression of loan_amount = .01 * loan_to_value_ratio * property_value, our RMSE is comparable to the RMSE found by most of the statistical models we tried. We saw that many of the more complex models struggled to pick up the relationship between property_value and loan_to_value_ratio, while for a statistician analyzing the data, it would be straightforward to pick up on this relationship. Even trying to create a linear model with just a few variables, including the interaction term property_value : loan_to_value yielded very similar results to our naive formula of calculating the final loan value.

# Overall Recommendations
Based on our experience working with this data, it is clear that creating a model around acceptances and denials is possible, but providing more training and testing data to this model could allow for better future results. Based on the previously provided figures, we can see our models learning general relationships between various covariates, such as credit score and payment to income ratio, to create a usable model for prediction, but some of the more specific covariates that could point to mortgage denials do not stand out within the sample, because of the limited number of denials available to train and test our models. <br>
### Assessment of our Models’ Performance
Looking at the various RMSE and accuracy metrics we had for our models, we are satisfied with the results of our modeling efforts. There are clear, logical, explainable trends found within the dataset, that we can model in various ways. For example, there is a clear relationship between having a high debt to income ratio and being denied for a mortgage. This is a logical relationship, considering that lenders would consider individuals with a higher debt to income ratio to have a greater risk of default, and would therefore deny their mortgage applications at a higher rate than the average Travis County resident. <br>
### Critique of our Methodology
In general, we placed a large emphasis on avoiding the overfitting problem with our finalized models. However, this is easier said than done, since the sheer fact of us hand-selecting a model based on test error is a factor in overfitting the trained models on our data selection. In other
words, even though each individual model only sees the training data, our model selection algorithms (highest accuracy and lowest RMSE on the testing data) are highly linked with the testing data set, which could cause an overfitting effect on the final selected model. There are not a lot of ways we can avoid this. <br>
### Future Assessments
Seeing that there is a large trove of data available, it would be interesting to do a more complete analysis on mortgage acceptance/denial rates between different races and ethnicities in the United States. If we want to derive any sort of causal inference from our dataset, it is a good opportunity to use matching in our dataset, to match individuals that have similar application details (i.e. credit score, debt to income ratio, property value, etc) and see if there are statistically significant differences in the acceptance or denial of their applications. With this information, and a deeper analysis into the various balance aspects of the matching process, we could derive interesting insights on mortgage denial/acceptance rates across races and ethnicities. <br>

Team: Nidhish Nerur, Sanath Govindarajan, Pavan Agrawal, Tanay Sethia
