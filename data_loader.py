''' First: preparing the data which is called from the main file.
Variables that might have effects on the survival are taken as:
  1) pclass = Ticket class
  2) Sex
  3) Age
  4) SibSp # Number of sibling in the ship
  5) Parch = # of parents/children aboard the Titanic
Will prepare the data assuming these 4 variables as dependent variables. '''

import numpy as np
import pandas as pd
import csv
import random

# Returns the data in required form from the raw data input
def load_train_data(filename):
	data_with_age = []
	data_without_age = []
	with open(filename, 'r') as file:
		try:
			reader = csv.reader(file, delimiter=',')
			# skip the header line of the train.csv file
			next(reader)
			no_pclass, no_sex, no_age, no_sibsp, no_parch, no_survived = 0,0,0,0,0,0
			sum_age = 0
			for row in reader:
				pclass, sex, age  = row[2], row[4], row[5]
				sibsp, parch, survived = row[6], row[7], row[1]
				
				# Giving correct result as a: y_ = 2 * 1 array 
				# where y_correct = [survived, died]
				# whichever (survived or died) is 1 (or larger) will be true.
				int_survived = int(survived)
				if int_survived == 1: y_correct = [1, 0]
				if int_survived == 0: y_correct = [0, 1]
					
				if sex == 'male': sex = 1
				if sex == 'female': sex = 0
				
				if age != "": sum_age += float(age)
				
				if age == "":
					pclass, sex  = int(pclass), int(sex), 
					sibsp, parch = int(sibsp), int(parch)
					data_without_age.append([[pclass, sex, age, sibsp, parch], y_correct])
				else:
					pclass, sex = int(pclass), int(sex)
					age, sibsp, parch = float(age), int(sibsp), int(parch)
					data_with_age.append([[pclass, sex, age, sibsp, parch], y_correct])

				if  pclass == "": no_pclass += 1
				if sex == "": no_sex += 1
				if age == "": no_age += 1
				if sibsp == "": no_sibsp += 1
				if parch == "": no_parch += 1
				if survived == "": no_survived += 1
				
			if no_pclass != 0 and no_sex != 0 and no_sibsp != 0 and no_survived != 0:
				print("There are some varibles other than age not defined in the train data!")
			
			# Replace no age data with average age
			average_age = sum_age/(len(data_with_age))
			for i in range(len(data_without_age)):
				data_without_age[i][0][2] = average_age
			# Combine both data lists
			joined_data = data_with_age + data_without_age
		finally:
			file.close()
			
		# Separating variables and survival into different lists 
		# and returning in the required form
		variables, survival = [], []
		for data in joined_data:
			variables.append(data[0])
			survival.append(data[1])
		return np.array(variables), np.array(survival)


# Using the pandas library - makes easy
# Returns the test data in the required input form
def load_test_data(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df, columns=["Pclass", "Sex", "Age", "SibSp", "Parch"])
    
    # Replacing the NaN age with average age of passengers
    average_age = df["Age"].mean()
    df["Age"].fillna(average_age, inplace=True)

    # Replacing Male: 1 and Female: 0
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    # Coverting and returning in the x_data/x_batch required form
    x_data = []
    x_matrix = df.as_matrix()
    for i in range(len(x_matrix)):
        x_data.append(x_matrix[i])
    return np.array(x_data)


# Returns the random batches of data of length n out of total data
def get_batch(xs_batch, ys_batch, n):
	random.shuffle(xs_batch)
	random.shuffle(ys_batch)
	x_batch, y_batch = xs_batch[0:n], ys_batch[0:n]
	return x_batch, y_batch
