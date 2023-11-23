import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import data
from pulp import *
import pickle

tasks = pd.read_csv('modified_data.csv')
resources = pd.read_csv('resources.csv')


print(tasks)


tasks['Start DateTime'] = pd.to_datetime(tasks.StartDate + " " + tasks.StartTime)

tasks = tasks.drop(['StartDate', 'StartTime'], axis=1)

tasks['End DateTime'] = pd.to_datetime(tasks.EndDate + " " + tasks.EndTime)

tasks = tasks.drop(['EndDate', 'EndTime'], axis=1)

compatabilities = {}

for index, row in tasks.iterrows():
    compatabilities[row.TaskId] = {}
    plane_types = resources[resources[row.AircraftTypeCode] == 1].ResourceId.values
    compatabilities[row.TaskId]['PlaneTypes'] = plane_types

print(compatabilities)

print(tasks)