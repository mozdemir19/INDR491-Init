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



