import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

target = 'y'
learning_rate = 0.04
max_depth = 8
n_estimators = 600

class Model:
    """
    This class represents a pipeline of training a model, making predictions and evaluating the performance of the pipeline
    """    
    def __init__(self, data: str="/path/to/csv", target: str=target):
        """This constructor imports the data and defines the predictors and the target variable

        Args:
            data (str): path to the csv file containing the training data. Defaults to "/path/to/csv".
            target (str): column name of the target variable. Defaults to target.

        Raises:
            Exception: target is not present in the data
        """
        # import the dataframe
        self.df = pd.read_csv(data)

        # Check to see if the target variable is present in the data
        if target not in self.df.columns:
            raise Exception(f"Target: {target} is not present in the data")
        
        # define the perdictors
        self.X = self.df.drop(target, axis=1)

        # define the target variable
        self.y = self.df[target]
    
    def train(self, learning_rate: float=learning_rate, max_depth: int=max_depth, n_estimators: int=n_estimators):
        """This function trains the model using k fold cross validation

        Args:
            learning_rate (float): Boosting learning rate for LGBMRegressor. Defaults to learning_rate.
            max_depth (int): Maximum tree depth for base learners, <=0 means no limit. Defaults to max_depth.
            n_estimators (int): Number of boosted trees to fit. Defaults to n_estimators.
        """
        print("training model...")
        self.model = LGBMRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        self.model.fit(self.X, self.y)
    
    def evaluate(self, X_test: pd.DataFrame=None, y_test: pd.Series=None):
        """This function evaluates the performance of the model

        Args:
            X_test (pd.DataFrame): the test set. Defaults to None.
            y_test (pd.Series): the target variable corresponding to the test set. Defaults to None.
        
        Raises:
            Exception: X_test and y_test have different number of observations
        """
        if X_test.shape[0] != y_test.shape[0]:
            raise Exception(f"Incorrect Dimension: X_test.shape[0] = {X_test.shape[0]} does not match y_test.shape[0] = {y_test.shape[0]}")
        y_pred = self.model.predict(X_test)

        self.results = pd.DataFrame({"MSE"  : [mean_squared_error(y_test, y_pred)],
                                     "RMSE" : [np.sqrt(mean_squared_error(y_test, y_pred))],
                                     "MAE"  : [mean_absolute_error(y_test, y_pred)],
                                     "R2"   : [r2_score(y_test, y_pred)]})
        print(self.results)
    
    def predict(self, X_test: pd.DataFrame=None):
        """This function makes predictions

        Args:
            X_test (pd.DataFrame): the test set. Defaults to None.

        Raises:
            Exception: test data and training data has different number of features
        """
        # Check to see if there is the same number of features as in the training data
        if X_test.shape[1] != self.X.shape[1]:
            raise Exception(f"data has different number of features from training data")
        # make the predictions
        self.predictions = self.model.predict(X_test)
        print(self.predictions)
    

def main():
    # create a pipeline object
    model = Model(data='train.csv', target='y')

    # train the model
    model.train()

    # import the test set
    test = pd.read_csv('test.csv')
    X_test = test.drop('y', axis=1)
    y_test = test['y']

    # evaluate the model
    model.evaluate(X_test, y_test)

    # make predictions
    model.predict(X_test.head())

if __name__ == '__main__':
    main()

