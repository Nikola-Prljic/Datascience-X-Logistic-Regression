# **Datascience-X-Logistic-Regression**

    The dataset contains Hogwarts houses and class grades.
    Using the grades, we will calculate which house a person belongs to.
    We will use multiple features for this prediction.
    We need a classification model with 4 possible outputs. The model should have a dense output layer with 4 nodes.

## **STEP 1**
- **Load the Data:** Load the dataset into a pandas DataFrame.
- **Handle Missing Values:** Remove or impute NaN values.
- **Encode Categorical Variables:** Convert house names to numeric values.
- **Select Features:** Choose the features for the model.

## **STEP 2**
- **Split Data:** Divide the dataset into training and testing sets.
- **Standardize Features:** Standardize the features to improve model performance.
- **Create the Model:** Define and compile the logistic regression model.
- **Train the Model:** Fit the model to the training data.
- **Evaluate the Model:** Evaluate the model on the testing set.

## It is possible to change the Features
You can use all Features or just 2.

Just change features_names list in app.py to what you want.

## **Dataset**

![Screenshot from 2024-05-24 10-37-07](https://github.com/Nikola-Prljic/Datascience-X-Logistic-Regression/assets/72382235/319d3381-be1a-4441-a9d6-22a7fd181f86)


# Virtualization with Tensorboard
### **RUN**
- python app.py
- tensorboard --logdir logs/fit
- open http://localhost:6006/ in your Browser

![TensorBoaard](https://github.com/Nikola-Prljic/Datascience-X-Logistic-Regression/assets/72382235/6a783da5-97ea-4934-8a76-a200bd09a2ba)
![graphs](https://github.com/Nikola-Prljic/Datascience-X-Logistic-Regression/assets/72382235/62fd0857-1f0e-48f6-9954-16ad8f30e291)
