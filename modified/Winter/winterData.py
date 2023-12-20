import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp as pl
from functools import reduce
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

data = pd.ExcelFile('RMS-MultiKPI-Input-KisDonemi-GercekVeri.xlsx')
tasks = pd.read_excel(data, sheet_name='Tasks')

tasks['StartDateTime'] = pd.to_datetime(tasks['StartDate'].astype(str) + ' ' + tasks['StartTime'].astype(str)) #pd.to_datetime(pd.to_datetime(tasks['StartDate']) + pd.to_datetime(tasks['StartTime']), format="%m-%d-%Y%H:%M:%S")
tasks['EndDateTime'] = pd.to_datetime(tasks['EndDate'].astype(str) + ' ' + tasks['EndTime'].astype(str))
resources = pd.read_excel(data, sheet_name='Resources')
serviceTypes = pd.read_excel(data, sheet_name='ServiceTypes')
serviceTypesDict = {}

for ind, row in serviceTypes.iterrows():
    if row['Code'] in ['J', 'C', 'D', 'P', 'A']:
        serviceTypesDict[row['Code']] = row['Name']

compatabilities = {}
for index, row in tasks.iterrows():
    compatabilities[row['TaskId']] = {}
    aircraftAndTaskType = resources[(resources[row['AircraftTypeCode'] + 'P'] == 1) & (resources[row['TaskTypeName']] == 1)].ResourceId.values.astype(str)
    if not pd.isna(row['ArrivalCategory']):
        arrivalCat = resources[resources[row['ArrivalCategory']] == 1].ResourceId.values.astype(str)
    if not pd.isna(row['DepartureCategory']):
        departureCat = resources[resources[row['DepartureCategory']] == 1].ResourceId.values.astype(str)
    if not pd.isna(row['ArrivalServiceType']):
        arrivalServiceType = resources[resources[row['ArrivalServiceType']] == 1].ResourceId.values.astype(str)
    if not pd.isna(row['DepartureServiceType']):
        departureServiceType = resources[resources[row['DepartureServiceType']] == 1].ResourceId.values.astype(str)
    
    compatabilities[row['TaskId']] = reduce(np.intersect1d, (aircraftAndTaskType, arrivalCat, departureCat, arrivalServiceType, departureServiceType)).tolist()
with open('winter_compat', 'w') as f:
    json.dump(compatabilities, f, indent=2)
time_series = pd.Series(True,
                        index= pd.date_range(start=tasks.StartDateTime.min()
                                             ,end=tasks.EndDateTime.max()
                                             ,freq=pd.offsets.Minute(1)))

def trunc_ts(series):
    return time_series.truncate(series['StartDateTime'], series['EndDateTime'])

"""taskHeatmap = tasks.apply(trunc_ts, axis=1).T
taskHeatmap[taskHeatmap == True] = 1
taskHeatmap = taskHeatmap.fillna(0).astype(int)
taskHeatmap.columns = tasks.TaskId.values
taskHeatmap = taskHeatmap.drop_duplicates()"""
taskHeatmap = pd.read_csv('heatmap.csv', index_col=0)
print(taskHeatmap)

taskList = tasks.TaskId.values
resourceList = resources.ResourceId.values

problem = pl.LpProblem('WinterData', pl.LpMaximize)

x = {}
U = {}

for t in taskList:
    for r in compatabilities[t]:
        x[t, r] = pl.LpVariable('x_%s,%s' % (t, r), lowBound=0, upBound=1, cat=pl.LpBinary)
        U[t, r] = 1

problem += pl.lpSum([U[var] * x[var] for var in x])

for t in taskList:
    problem += pl.lpSum([x[t, r] for r in compatabilities[t]]) <= 1

for idx, row in taskHeatmap.iterrows():
    tasks_in_time_step = set(dict(row[row==1]).keys())
    for r in resourceList:
        cons = [x[t, r] for t in tasks_in_time_step if (t, r) in x]
        if len(cons) > 1:
            constraint_for_time_bucket = pl.lpSum(cons) <= 1
            problem += constraint_for_time_bucket

problem.solve()

assignments = pd.DataFrame(columns=['TaskId', 'ResourceId'])
for var in x:
    #print(x[var].varValue)
    if x[var].varValue == 1:
        
        #new_row = pd.DataFrame({'TaskId': var[0], 'ResourceId':var[1]})
        assignments.loc[len(assignments)] = [var[0], var[1]]

gantt_df = pd.DataFrame({'ResourceId': assignments.ResourceId.values, 'TaskId': assignments.TaskId.values, 
                         #'Duration': (tasks.loc[assignments.TaskId.values - 1].End_DateTime - tasks.loc[assignments.TaskId.values - 1].Start_DateTime).values,
                         'StartDateTime': tasks.loc[assignments.TaskId.values - 1].StartDateTime.values,
                         'EndDateTime': tasks.loc[assignments.TaskId.values - 1].EndDateTime.values})


fig = px.timeline(gantt_df, x_start='StartDateTime', x_end='EndDateTime', y='ResourceId', color='TaskId')
fig.show()

