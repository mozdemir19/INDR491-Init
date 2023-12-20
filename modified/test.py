import pandas as pd
import numpy as np
import pulp as pl
import plotly.express as px

gantt = pd.read_csv('Summer/assignments.csv')
tasks = pd.read_excel('Summer/RMS-MultiKPI-Input-YazDonemi-GercekVeri.xlsx', sheet_name='Tasks')


tasks['StartDateTime'] = pd.to_datetime(tasks['StartDate'].astype(str) + ' ' + tasks['StartTime'].astype(str)) 
tasks['EndDateTime'] = pd.to_datetime(tasks['EndDate'].astype(str) + ' ' + tasks['EndTime'].astype(str))


gantt = gantt[['TaskId', 'ResourceId']]
gantt['StartDateTime'] = tasks.loc[gantt.TaskId - 1].StartDateTime
gantt['EndDateTime'] = tasks.loc[gantt.TaskId - 1].EndDateTime



fig = px.timeline(gantt, x_start='StartDateTime', x_end='EndDateTime', y='ResourceId', color='TaskId', color_continuous_scale='rainbow')
fig.show()