'''
First: preparing the data which is called from the main file.
Variables that might have effects on the survival are taken as:
  1) pclass = Ticket class
  2) Sex 
  3) Age
  4) parch = # of parents/children aboard the Titanic  
Will prepare the data assuming these 4 variables as dependent variables.  
'''
import numpy as np
import csv
import random

# Returns the data in required form from the raw data input
def data_loader(filename):
	train_data_with_age = []
	train_data_without_age = []
	with open(filename, 'r') as file:
		try:
			reader = csv.reader(file, delimiter=',')
			next(reader)
			no_pclass, no_sex, no_age, no_parch, no_survived = 0,0,0,0,0
			for row in reader:
				pclass, sex, age, parch, survived = row[2], row[4], row[5], row[7], row[1]
		
				if sex == 'male': sex = 1
				if sex == 'female': sex = 0
		
				if age == "":
					train_data_without_age.append([[int(pclass), int(sex), age, int(parch)], [int(survived)]])
				else:
					train_data_with_age.append([[int(pclass), int(sex), float(age), int(parch)], [int(survived)]])
			
				if  pclass == "": no_pclass += 1
				if sex == "": no_sex += 1
				if age == "": no_age += 1
				if parch == "": no_parch += 1
				if survived == "": no_survived += 1
		finally:
			file.close()
		'''		
		print("Total passenger with no pclass: ", no_pclass)
		print("Total passenger with no sex: ", no_sex)
		print("Total passenger with no age: ", no_age)
		print("Total passenger with no parch: ", no_parch)
		print("Total passenger with no survived: ", no_survived)
		'''
		return train_data_with_age, train_data_without_age

# Returns the random batches of data out of total data 
def get_batch(xs_batch, ys_batch, n):
	random.shuffle(xs_batch)
	random.shuffle(ys_batch)
	x_batch, y_batch = xs_batch[0:n], ys_batch[0:n]
	return x_batch, y_batch
	
	








		
