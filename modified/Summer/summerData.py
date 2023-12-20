import sys
sys.path.append('/home/memo/Desktop/Dersler/INDR491/first_model/modified')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp as pl
import plotly.express as px
import plotly.graph_objects as go
import utils as util
import os.path

priorityScores = [500, 400, 300, 200]
min_bucket = 1
winter = '/../Winter/RMS-MultiKPI-Input-KisDonemi-GercekVeri.xlsx'
summer = '/RMS-MultiKPI-Input-YazDonemi-GercekVeri.xlsx'
dataFilePath = os.path.dirname(__file__) + summer

data = pd.ExcelFile(dataFilePath)
tasks = pd.read_excel(data, sheet_name='Tasks')

resources = pd.read_excel(data, sheet_name='Resources')
priorityContent = pd.read_excel(data, sheet_name='PriorityContents')

tasks['StartDateTime'] = pd.to_datetime(tasks['StartDate'].astype(str) + ' ' + tasks['StartTime'].astype(str)) 
tasks['EndDateTime'] = pd.to_datetime(tasks['EndDate'].astype(str) + ' ' + tasks['EndTime'].astype(str))

tasks['EndWithBuffer'] = tasks['EndDateTime'] + pd.offsets.Minute(15)

compatibilities = util.compute_compatabilities(tasks, resources)

priorities = util.read_prios(tasks, resources, priorityContent, priorityScores)

taskHeatmap = util.heatmap(tasks)

taskList = tasks.TaskId
resourceList = resources.ResourceId

problem = pl.LpProblem('SummerData', pl.LpMaximize)

x = {}
U = {}

for t in taskList:
    for r in resourceList:
        x[t, r] = pl.LpVariable('x_%s,%s' % (t, r), lowBound=0, upBound=compatibilities.loc[t, r], cat=pl.LpBinary)
        U[t, r] = 1

problem += pl.lpSum([(U[var] + np.sum([priorities[i].loc[var] for i in range(len(priorities))])) * x[var] for var in x])

#CONSTRAINTS

#i) assign one task to one resource
for t in taskList:
    problem += pl.lpSum([x[t, r] for r in resourceList]) <= 1


#ii) assign one resource to one task at a time
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


print(assignments)
assignments.to_csv('assignments.csv')
##print gantt chart
gantt_df = pd.DataFrame({'ResourceId': assignments.ResourceId, 'TaskId': assignments.TaskId, 
                         #'Duration': (tasks.loc[assignments.TaskId.values - 1].End_DateTime - tasks.loc[assignments.TaskId.values - 1].Start_DateTime).values,
                         'StartDateTime': tasks.loc[assignments.TaskId - 1].StartDateTime,
                         'EndDateTime': tasks.loc[assignments.TaskId - 1].EndDateTime})


fig = px.timeline(gantt_df, x_start='StartDateTime', x_end='EndDateTime', y='ResourceId', color='TaskId', color_continuous_scale='rainbow')
fig.show()

