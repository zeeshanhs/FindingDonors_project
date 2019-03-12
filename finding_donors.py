import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.tree import DecisionTreeRegressor
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from IPython.display import display # To display entire dataset


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit




# print(np.linspace(.1, 1.0, 5))

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head())

# TODO: Total number of records
n_records = data['age'].count()

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data['age'][data['income'] != '<=50K'].count()

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data['age'][data['income'] == '<=50K'].count()

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = None

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))