import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp as pl
from functools import reduce
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go


min_bucket=1

tasks = pd.read_csv('modified_data.csv')
resources = pd.read_csv('resources.csv')
serviceTypeDict = {'J': 'Normal_Service',
                   'A': 'Cargo/Mail'}

#print(tasks)

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
print(tasksHeatmap)


task_list = tasks.TaskId.values
resource_list = resources.ResourceId.values

priorityGroupOne = tasks
priorityGroupTwoOneList = priorityGroupOne.TaskId.values
print(priorityGroupOne)
priorityGroupTwo = tasks[tasks.TaskTypeName.isin(['Single Paired Task', 'Sadece Arrival', 'Sadece Departure'])]
priorityGroupTwoTaskList = priorityGroupTwo.TaskId.values
print(priorityGroupTwo)
priorityGroupThree = tasks[tasks['AircraftTypeCode'] == 'F']
priorityGroupThreeTaskList = priorityGroupThree.TaskId.values
print(priorityGroupThree)

problem = pl.LpProblem('Adnan_Menderes', pl.LpMaximize)

x = {}
U = {}
np.random.seed(1234)

### DEFINE VARIABLES, only for compatible task-resource pairs
for t in task_list:
    for r in compatabilities[t]:
        x[t, r] = pl.LpVariable("x_%i,%s" % (t, r), lowBound=0, upBound=1, cat=pl.LpBinary)
        U[t, r] = 1 #np.random.randint(1, 10)

# this variable is used to introduce constants to the objective function
constOne = pl.LpVariable('constantOne', lowBound=1, upBound=1)


print(priorityGroupThree.shape[0] - pl.lpSum([x[(t, r)] for t in priorityGroupThreeTaskList for r in compatabilities[t]]))

### DEFINE OBJECTIVE FUNCTION, over existing variables
problem += pl.lpSum([pl.lpSum([U[var] * x[var] for var in x]), 
            -0.5 * (priorityGroupThree.shape[0] * constOne - pl.lpSum([x[(t, r)] for t in priorityGroupThreeTaskList for r in compatabilities[t]])), 
            -0.5 * (priorityGroupTwo.shape[0] * constOne - pl.lpSum([x[(t, r)] for t in priorityGroupTwoTaskList for r in compatabilities[t]])),
            #-0.5 * (priorityGroupOne.shape[0] * constOne)
            ])

### DEFINE CONSTRAINTS
# i) each task is assigned to maximum of one resource
for t in task_list:
    if len(compatabilities[t]) > 0:
        problem += pl.lpSum([x[t, r] for r in compatabilities[t]]) <= 1
        #print(pl.lpSum([x[t, r] for r in compatabilities[t]]))
    

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
            #print(pl.lpSum(cons))
            #print(lpSum(cons))
            constraint_for_time_bucket = pl.lpSum(cons) <= 1
            # These will occur when the plane overlaps change
            problem += constraint_for_time_bucket

# iii) assign 

problem.solve(pl.PULP_CBC_CMD(msg=1))

total = 0
assignments = pd.DataFrame(columns=['TaskId', 'ResourceId'])
for var in x:
    #print(x[var].varValue)
    if x[var].varValue == 1:
        
        #new_row = pd.DataFrame({'TaskId': var[0], 'ResourceId':var[1]})
        assignments.loc[len(assignments)] = [var[0], var[1]]
        total += 1
        #print(x[var].varValue)
        #print(var[0], var[1])
    

print(priorityGroupThree.shape[0] - pl.lpSum([x[(t, r)].varValue for t in priorityGroupThreeTaskList for r in compatabilities[t]]))
print(priorityGroupTwo.shape[0] - pl.lpSum([x[(t, r)].varValue for t in priorityGroupTwoTaskList for r in compatabilities[t]]))
#print(assignments)
assignments.to_csv('assignments.csv', index=False)
lencompat = 0
for var in compatabilities:
    if len(compatabilities[var]) > 0:
        lencompat += 1


gantt_df = pd.DataFrame({'ResourceId': assignments.ResourceId.values, 'TaskId': assignments.TaskId.values, 
                         #'Duration': (tasks.loc[assignments.TaskId.values - 1].End_DateTime - tasks.loc[assignments.TaskId.values - 1].Start_DateTime).values,
                         'StartDateTime': tasks.loc[assignments.TaskId.values - 1].Start_DateTime.values,
                         'EndDateTime': tasks.loc[assignments.TaskId.values - 1].End_DateTime.values})

print(problem.objective)
fig = px.timeline(gantt_df, x_start='StartDateTime', x_end='EndDateTime', y='ResourceId', color='TaskId')
fig.show()
