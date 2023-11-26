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

print(compatabilities)
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
tasksHeatmap = tasksHeatmap.drop_duplicates()


task_list = tasks.TaskId.values
resource_list = resources.ResourceId.values

problem = LpProblem('Adnan_Menderes', LpMaximize)

x = {}
U = {}
np.random.seed(1234)

### DEFINE VARIABLES, only for compatible task-resource pairs
for t in task_list:
    for r in compatabilities[t]:
        x[t, r] = LpVariable("t%i_r%s" % (t, r), lowBound=0, upBound=1, cat=LpBinary)
        U[t, r] = np.random.randint(1, 10)
        
### DEFINE OBJECTIVE FUNCTION, over existing variables
problem += lpSum([U[var] * x[var] for var in x])

### DEFINE CONSTRAINTS
# i) each task is assigned to maximum of one resource
for t in task_list:
    if len(compatabilities[t]) > 0:
        problem += lpSum([x[t, r] for r in compatabilities[t]]) <= 1
        print(lpSum([x[t, r] for r in compatabilities[t]]))
    

# ii) assign only one task to a resource per time-step
for idx, row in tasksHeatmap.iterrows():
    # Get all the tasks in time-step
    tasks_in_time_step = set(dict(row[row==1]).keys())
    #print(tasks_in_time_step)
    # For all gates
    for r in resource_list:
        # Constraints may be blank
        cons = [x[t, r] for t in tasks_in_time_step if (t, r) in x]
        
        # Only need to impose constraint if there is an overlap
        if len(cons) > 1:
            print(lpSum(cons))
            #print(lpSum(cons))
            constraint_for_time_bucket = lpSum(cons) <= 1
            # These will occur when the plane overlaps change
            problem += constraint_for_time_bucket

problem.solve()

total = 0
print('TaskId', '\t', 'ResourceId')
assignments = pd.DataFrame(columns=['TaskId', 'ResourceId'])
for var in x:
    #print(x[var].varValue)
    if x[var].varValue == 1:
        
        #new_row = pd.DataFrame({'TaskId': var[0], 'ResourceId':var[1]})
        assignments.loc[len(assignments)] = [var[0], var[1]]
        total += 1
        #print(x[var].varValue)
        #print(var[0], var[1])
    


print(assignments)
print(total)
lencompat = 0
for var in compatabilities:
    if len(compatabilities[var]) > 0:
        lencompat += 1

print(lencompat)


#print(compatabilities)

#print(tasks)