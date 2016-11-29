#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import os

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# Calculate POI count directly
poi_cnt = reduce(lambda x,y: x+y, [enron_data[person_name]["poi"] for person_name in enron_data.keys()])
# The last name POI List(POT is True)
poi_names_in_data = [person_name.split(' ')[0] for person_name in enron_data.keys() if enron_data[person_name]["poi"] is True]

# Open poi_names.txt to read all the POI names into list
'''
poi_names = []
poi_names_file = open('../final_project/poi_names.txt', 'r')
for idx, line in enumerate(poi_names_file):
    if idx in (0, 1):
        continue
    else:
        last_name = line.split(' ')[1]
        last_name = last_name.split(',')[0]
        poi_names.append(last_name.upper())
poi_names_file.close()
'''

# print(sorted(poi_names_in_data))
# print(sorted(poi_names))

# Find out James Prentice's total stock value'
print(enron_data['PRENTICE JAMES']['total_stock_value'])
# >>> 1095040

# Find out Wesley Colwell's from_this_person_to_poi
print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
# >>> 11

# Find out Jeffrey K Skilling's stock options exercised
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])
# >>> 19250000

# Find out who(Lay, Skilling, Fastow) took home the most money
chair_men = [enron_data[name] for name in enron_data if name.split(' ')[0] in ('LAY', 'SKILLING', 'FASTOW')]
max_payment_in_chair_men = max(chair_men, key=lambda x: x['total_payments'])
print(max_payment_in_chair_men['email_address'], max_payment_in_chair_men['total_payments'])
# >>> ('kenneth.lay@enron.com', 103559793)

# How many folks in this dataset have a quantified salary ?
# Known email address
# 'salary' 'email_address'
folk_have_salary = len([name for name in enron_data if enron_data[name]['salary'] != 'NaN'])
folk_have_email = len([name for name in enron_data if enron_data[name]['email_address'] != 'NaN'])
print(folk_have_salary)
# >>> 95
print(folk_have_email)
# >>> 111

# What percentage of people in the dataset have "NaN" for their total payments?
people_have_no_payment = len([name for name in enron_data if enron_data[name]['total_payments'] == 'NaN'])
total_people = len(enron_data)
print(float(people_have_no_payment)/float(total_people))
# >>> 0.143835616438

# What percentage of POIs in the dataste have "NaN" for their payments?
poi_have_no_payment = len([name for name in enron_data if 
                           enron_data[name]["poi"] is True and
                           enron_data[name]['total_payments'] == 'NaN'])

print(float(poi_have_no_payment)/float(poi_cnt))
# >>> 0.0

# Summary:
# Because all the POI list on financial spreadsheet and have total payment
# If you use these data to train a POI detector to identify POI
# when a POI have not list in financial spreadsheet but it is indeed a POI
# this model will have bias and mistake to classify this case to non poi
# So you can add some POI and hvae NaN in total payment to balance the dataset 