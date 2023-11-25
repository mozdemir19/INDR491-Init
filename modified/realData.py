import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import *
from functools import reduce
import pickle

min_bucket=1



tasks = pd.read_csv('modified_data.csv')
resources = pd.read_csv('resources.csv')
serviceTypeDict = {'J': 'Normal_Service',
                   'A': 'Cargo/Mail'}

print(tasks)


tasks['Start_DateTime'] = pd.to_datetime(tasks.StartDate + " " + tasks.StartTime)

tasks = tasks.drop(['StartDate', 'StartTime'], axis=1)

tasks['End_DateTime'] = pd.to_datetime(tasks.EndDate + " " + tasks.EndTime)

tasks = tasks.drop(['EndDate', 'EndTime'], axis=1)

compatabilities = {}

for index, row in tasks.iterrows():
    compatabilities[row.TaskId] = {}
    
    aircraftAndTaskType = resources[(resources[row.AircraftTypeCode] == 1) & (resources[row.TaskTypeName] == 1)].ResourceId.values
    if not pd.isna(row.ArrivalCategory):
        arrivalCat = resources[(resources[row.ArrivalCategory] == 1)].ResourceId.values
    if not pd.isna(row.DepartureCategory):
        departureCat = resources[(resources[row.DepartureCategory] == 1)].ResourceId.values
    if not pd.isna(row.ArrivalServiceType):
        arrivalService = resources[(resources[serviceTypeDict[row.ArrivalServiceType]] == 1)].ResourceId.values
    if not pd.isna(row.DepartureServiceType):
        departureService = resources[(resources[serviceTypeDict[row.DepartureServiceType]] == 1)].ResourceId.values
    
    #resourceArrays = [aircraftAndTaskType, arrivalCat, departureCat, arrivalService, departureService]
    compatabilities[row.TaskId] = reduce(np.intersect1d, (aircraftAndTaskType, arrivalCat, departureCat, arrivalService, departureService)).tolist()


time_series = pd.Series(True, index= pd.date_range(
        start=tasks.Start_DateTime.min(),
        end=tasks.End_DateTime.max(),
        freq=pd.offsets.Minute(min_bucket)))

def trunc_ts(series):
    return time_series.truncate(series['Start_DateTime'], series['End_DateTime'])

tasksHeatmap = tasks.apply(trunc_ts, axis=1).T

tasksHeatmap[tasksHeatmap == True] = 1
tasksHeatmap = tasksHeatmap.fillna(0).astype(int)
tasksHeatmap.columns = tasks.TaskId.values

tasksHeatmap['tot'] = tasksHeatmap.sum(axis=1)
tasksHeatmap = tasksHeatmap[tasksHeatmap.tot > 1]
tasksHeatmap.drop(['tot'], axis=1, inplace=True)

tasksHeatmap = tasksHeatmap.drop_duplicates()


task_list = tasks.TaskId.values
resource_list = resources.ResourceId.values

problem = LpProblem('Adnan_Menderes', LpMaximize)

x = {}
U = {}

for t in task_list:
    for r in compatabilities[t]:
        x[t, r] = LpVariable("t%i_r%s" % (t, r), 0, 1, LpBinary)
        U[t, r] = np.random.randint(1, 10)
        
problem += lpSum([U[t, r] * x[t, r] for t in task_list for r in compatabilities[t]])

   
for idx, row in tasksHeatmap.iterrows():
    # Get all the turns for time-bucket
    turns_in_time_bucket = set(dict(row[row==1]).keys())
    # For all gates
    for r in resource_list:
        # Constraints may be blank
        cons = [x[t, r] for t in turns_in_time_bucket if (t, r) in x]
        # Only need to impose constraint if there is an overlap
        if len(cons) > 1:
            constraint_for_time_bucket = lpSum(cons) <= 1
            # These will occur when the plane overlaps change
            problem += constraint_for_time_bucket
problem.solve()



#print(compatabilities)

#print(tasks)