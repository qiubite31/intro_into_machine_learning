#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL') 
data = featureFormat(data_dict, features)

### your code below

maximum = 0
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

    if bonus > maximum:
        maximum = bonus

# find outlier I, and the outlier is TOTAL
# print maximum
# for record, value in data_dict.items():
#     if value['bonus'] == maximum:
#         print record 

# find outlier II, find the bonus > 5000000
for record, value in data_dict.items():
     if value['bonus'] > 5000000 and value['salary'] > 1000000 and value['bonus'] != 'NaN':
         print record, value['salary'], value['bonus']


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
