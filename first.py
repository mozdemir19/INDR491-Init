import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import *
import pickle

turns = pd.read_csv('turns.csv')
print(turns.head())
airport = pd.read_csv('airport.csv')
print(airport.head())


compatible_gates = {}
for idx, row in turns.iterrows():
    gates_lst = airport[airport.max_size >= row.plane_size].gate.values
    compatible_gates[row.turn_no] = gates_lst

print(compatible_gates)

# Create time-series between arrival of first plane and departure of last

min_bucket=5
time_series = pd.Series(True, index= pd.date_range(
        start=turns.inbound_arrival.min(),
        end=turns.outbound_departure.max(),
        freq=pd.offsets.Minute(min_bucket)))
    
# Truncate full time-series to [inbound_arrival, outbound_departure]
def trunc_ts(series):
    return time_series.truncate(series['inbound_arrival'], series['outbound_departure'])
    
heatmapdf = turns.apply(trunc_ts, axis=1).T
    
# Convert columns from index to turn_no
heatmapdf.columns = turns['turn_no'].values
# Cast to integer
heatmapdf = heatmapdf.fillna(0).astype(int)
heatmapdf.index = heatmapdf.index.time

#print(heatmapdf.head())


heatmapdf['tot'] = heatmapdf.sum(axis=1)
heatmapdf = heatmapdf[heatmapdf.tot > 1]
heatmapdf.drop(['tot'], axis=1, inplace=True)

heatmapdf = heatmapdf.drop_duplicates()

print(heatmapdf.shape)
heatmapdf.to_csv("time_preiods.csv")

turn_list = turns.turn_no.values
gate_list = airport.gate.values

problem = LpProblem('Adnan_Menderes', LpMaximize)


####Define x_i,j to be binary turn i assigned to gate j
x = {}
U = {}
for t in turn_list:
    for g in compatible_gates[t]:
        x[t, g] = LpVariable("t%i_g%s" % (t, g), 0, 1, LpBinary)
        U[t, g] = 1


#### Objective function sum over U * x
problem += lpSum([U[t, g] * x[t, g] for t in turn_list for g in compatible_gates[t]])





#### Constraints
## i) Each turn gets one gate
for t in turn_list:
    problem += lpSum(x[t, g] for g in gate_list if (t, g) in x) == 1


## ii) 
for idx, row in heatmapdf.iterrows():
    # Get all the turns for time-bucket
    turns_in_time_bucket = set(dict(row[row==1]).keys())
    # For all gates
    for g in gate_list:
        # Constraints may be blank
        cons = [x[t, g] for t in turns_in_time_bucket if (t, g) in x]
        # Only need to impose constraint if there is an overlap
        if len(cons) > 1:
            constraint_for_time_bucket = lpSum(cons) <= 1
            # These will occur when the plane overlaps change
            problem += constraint_for_time_bucket


problem.solve()

print("Status: ", LpStatus[problem.status])
print("Maximised Util.: ", value(problem.objective))


def plot_gantt_chart(allocated_turns, lp_variable_outcomes, min_bucket=5):
    
    # Assign gate
    for alloc in lp_variable_outcomes:
        if lp_variable_outcomes[alloc].varValue:
            allocated_turns.loc[allocated_turns['turn_no'] == alloc[0], 'gate'] = alloc[-1]

                
    # Create time-series between arrival of first plane and departure of last
    time_series = pd.Series(True, index= pd.date_range(
            start=turns.inbound_arrival.min(),
            end=turns.outbound_departure.max(),
            freq=pd.offsets.Minute(min_bucket)))

    # Truncate full time-series to [inbound_arrival, outbound_departure]
    def trunc_ts(series):
        return time_series.truncate(series['inbound_arrival'], series['outbound_departure'])

    # Allocations heat-map
    allocheatmapdf = allocated_turns.apply(trunc_ts, axis=1).T
    allocheatmapdf.columns = allocated_turns['turn_no'].values
    allocheatmapdf = allocheatmapdf.fillna(0).astype(int)
    allocheatmapdf.index = allocheatmapdf.index.time
    
    # Replace values with col-names
    for col in list(allocheatmapdf.columns):
        allocheatmapdf.loc[allocheatmapdf[col] > 0, col] = col
          
    # Columns are now stands
    allocheatmapdf.columns = allocated_turns['gate'].values  
    trans = allocheatmapdf.T

    # These will never overlap given the constraints
    plt_df = trans.groupby(trans.index).sum()

    # Plot
    sns.set()
    plt.figure(figsize=(20, 10))
    g = sns.heatmap(plt_df, xticklabels=10, cmap='nipy_spectral')
    plt.show()

#for alloc in x:
#    print(x[alloc].varValue) if x[alloc].varValue == 1 else print()
plot_gantt_chart(allocated_turns=turns, lp_variable_outcomes=x)