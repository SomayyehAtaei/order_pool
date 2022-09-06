import pandas as pd
from sklearn.model_selection import GridSearchCV
import sklearn.cluster as sc
import numpy as np
from geopy import distance
from sklearn.cluster import DBSCAN
import warnings
from prettytable import PrettyTable
from functools import reduce
from operator import concat

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pause_plot_time = 100

prepending_time = 30

# Calculate delivery cost for a batch
def calc_batch_cost(group_batch_distances, group_batch_size, terminal_fare):
    base_fare = 7
    km_fare = [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    return base_fare + sum(km_fare[: int(np.ceil(group_batch_distances))]) + (terminal_fare * (group_batch_size-1))

# Calculate delivery time for every tierminal in a batch
def calc_batch_time(group_batch_dist):
    ack_time = 7
    ack_to_arrive = 3
    waiting_source = 3
    km_time = 4
    wainting_destination = 2

    batch_count = len(group_batch_dist)
    order_time = []

    for i in range(batch_count):
        order_time.append(round(ack_time + ack_to_arrive + (batch_count * waiting_source) + (sum(group_batch_dist[:i+1]) * km_time) + (i * wainting_destination), 3))

    return order_time

# Calculate distance for each batch in a group with respect to shortest path
def calc_dist(group, terminal_fare):
    batch_count = np.max(group['batch']+1)
    supermarket = (group.iloc[0]['source_latitude'], group.iloc[0]['source_longitude'])

    group_batch_distances = np.zeros(int(batch_count)) # list of distances sumation for all batches
    group_batch_cost = np.zeros(int(batch_count)) # list of cost for all batch
    group_batch_size = np.zeros(int(batch_count))
    group_batch_path = []
    group_batch_dist = []
    group_batch_time = []

    for b in range(int(batch_count)):
        batch = group[group['batch'] == b]
        graph = make_batch_graph(batch)

        distance_to_sm = []
        for j, r in batch.iterrows():
            distance_to_sm.append(distance.distance(supermarket, (r['destination_latitude'], r['destination_longitude'])).kilometers)

        path = []
        batch_dist = []
        seen = np.zeros(batch.shape[0])

        if batch.shape[0] == 1:
            path.append(batch.iloc[0]['order_id'])
            batch_dist.append(distance_to_sm)

        else:
            i = np.argmin(distance_to_sm)
            path.append(batch.iloc[i]['order_id'])
            batch_dist.append(distance_to_sm[i])
            seen[i] = 1

            for k in range(batch.shape[0] - 1):
                sorted_index = np.argsort(graph[i][:])
                s = 1
                while(seen[sorted_index[s]]==1):
                    s = s+1

                path.append(batch.iloc[sorted_index[s]]['order_id'])
                batch_dist.append(graph[i][sorted_index[s]])
                seen[sorted_index[s]] = 1

        group_batch_distances[b] = np.sum(batch_dist)
        group_batch_dist.append(batch_dist)
        group_batch_size[b] = batch.shape[0]
        group_batch_cost[b] = calc_batch_cost(group_batch_distances[b], group_batch_size[b], terminal_fare)

        if len(batch_dist) > 1:
            group_batch_time.append(calc_batch_time(batch_dist))
        else:
            group_batch_time.append(calc_batch_time(batch_dist[0]))

        group_batch_path.append(path)

    group_max_dist = np.max(group_batch_distances)
    return group_batch_distances, group_batch_size, group_batch_cost, group_batch_time, group_max_dist, group_batch_path, group_batch_dist

# Calculate order time range in a 24h
def order_time_coding(data, prepending_time):

    data['date'] = data['first_created_at'].dt.date
    data['hour'] = data['first_created_at'].dt.hour
    data['minute'] = data['first_created_at'].dt.minute
    data['order_time_code'] = ((data['hour'] * 60) + data['minute']) // prepending_time

    data = data.drop(['hour', 'minute'], axis=1)
    return data

# Calculate distance for a batch
def make_batch_graph(batch):
    n = batch.shape[0]
    graph = np.zeros((n, n))

    for k in range(n):
        for j in range(n):
            distance_pair = distance.distance((batch.iloc[k]['destination_latitude'], batch.iloc[k]['destination_longitude']),
                                              (batch.iloc[j]['destination_latitude'], batch.iloc[j]['destination_longitude'])).kilometers
            graph[k,j] = round(distance_pair, 2)

    return graph

# Apply clusterng algorithm on each group and choose the best results
def clustering_group(group, terminal_fare, batch_size, time_slot):
    points = pd.concat((group['destination_latitude'], group['destination_longitude']), axis=1)
    points.rename(columns={'destination_latitude': 'latitude', 'destination_longitude': 'longitude'}, inplace=True)

    group_db = group
    group_km = group

    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian
    group_db['batch'] = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit_predict(np.radians(points))

    group_batch_distances_db, group_batch_size_db, group_batch_cost_db, \
    group_batch_time_db, group_max_dist_db, group_batch_path_db, group_batch_dist_db = calc_dist(group_db, terminal_fare)

#########################################################################################################
    km_model = sc.KMeans()
    param_grid = {"n_clusters": range((group.shape[0] // batch_size)+1, group.shape[0]+1)}
    search = GridSearchCV(km_model, param_grid=param_grid)
    km_model = search.fit(np.radians(points))
    group_km['batch'] = km_model.predict(np.radians(points))

    group_batch_distances_km, group_batch_size_km, group_batch_cost_km, \
    group_batch_time_km, group_max_dist_km, group_batch_path_km, group_batch_dist_km = calc_dist(group_km, terminal_fare)

#########################################################################################################
    max_time_km = max(reduce(concat, group_batch_time_km))
    max_time_db = max(reduce(concat, group_batch_time_db))

    if (max_time_km <= time_slot) and (max_time_db <= time_slot):#in both clustering all orders would be on time
        if sum(group_batch_cost_km) > sum(group_batch_cost_db):
            for b in range(len(group_batch_size_db)):
                u = 0
                c = round(group_batch_cost_db[b] / group_batch_size_db[b], 3)
                for t in range(int(group_batch_size_db[b])):
                    group_db.loc[group_batch_path_db[b][t], 'terminal'] = u
                    group_db.loc[group_batch_path_db[b][t], 'avg_cost'] = c
                    group_db.loc[group_batch_path_db[b][t], 'delivery_time'] = group_batch_time_db[b][t]
                    u = u + 1
            group = group_db

        else:
            for b in range(len(group_batch_size_km)):
                u = 0
                c = round(group_batch_cost_km[b] / group_batch_size_km[b], 3)
                for t in range(int(group_batch_size_km[b])):
                    group_km.loc[group_batch_path_km[b][t], 'terminal'] = u
                    group_km.loc[group_batch_path_km[b][t], 'avg_cost'] = c
                    group_km.loc[group_batch_path_km[b][t], 'delivery_time'] = group_batch_time_km[b][t]
                    u = u + 1
            group = group_km

    elif (max_time_km > time_slot) and (max_time_db > time_slot):#in both clustering some orders would be late
        if max_time_km > max_time_db:
            for b in range(len(group_batch_size_db)):
                u = 0
                c = round(group_batch_cost_db[b] / group_batch_size_db[b], 3)
                for t in range(int(group_batch_size_db[b])):
                    group_db.loc[group_batch_path_db[b][t], 'terminal'] = u
                    group_db.loc[group_batch_path_db[b][t], 'avg_cost'] = c
                    group_db.loc[group_batch_path_db[b][t], 'delivery_time'] = group_batch_time_db[b][t]
                    u = u + 1
            group = group_db

        elif max_time_km < max_time_db:
            for b in range(len(group_batch_size_km)):
                u = 0
                c = round(group_batch_cost_km[b] / group_batch_size_km[b], 3)
                for t in range(int(group_batch_size_km[b])):
                    group_km.loc[group_batch_path_km[b][t], 'terminal'] = u
                    group_km.loc[group_batch_path_km[b][t], 'avg_cost'] = c
                    group_km.loc[group_batch_path_km[b][t], 'delivery_time'] = group_batch_time_km[b][t]
                    u = u + 1
            group = group_km

        elif max_time_km == max_time_db:
            if sum(group_batch_cost_km) > sum(group_batch_cost_db):
                for b in range(len(group_batch_size_db)):
                    u = 0
                    c = round(group_batch_cost_db[b] / group_batch_size_db[b], 3)
                    for t in range(int(group_batch_size_db[b])):
                        group_db.loc[group_batch_path_db[b][t], 'terminal'] = u
                        group_db.loc[group_batch_path_db[b][t], 'avg_cost'] = c
                        group_db.loc[group_batch_path_db[b][t], 'delivery_time'] = group_batch_time_db[b][t]
                        u = u + 1
                group = group_db

            else:
                for b in range(len(group_batch_size_km)):
                    u = 0
                    c = round(group_batch_cost_km[b] / group_batch_size_km[b], 3)
                    for t in range(int(group_batch_size_km[b])):
                        group_km.loc[group_batch_path_km[b][t], 'terminal'] = u
                        group_km.loc[group_batch_path_km[b][t], 'avg_cost'] = c
                        group_km.loc[group_batch_path_km[b][t], 'delivery_time'] = group_batch_time_km[b][t]
                        u = u + 1
                group = group_km

    elif (max_time_km > time_slot) and (max_time_db < time_slot):#choose the clustering such that all orders would be on time
        for b in range(len(group_batch_size_db)):
            u = 0
            c = round(group_batch_cost_db[b] / group_batch_size_db[b], 3)
            for t in range(int(group_batch_size_db[b])):
                group_db.loc[group_batch_path_db[b][t], 'terminal'] = u
                group_db.loc[group_batch_path_db[b][t], 'avg_cost'] = c
                group_db.loc[group_batch_path_db[b][t], 'delivery_time'] = group_batch_time_db[b][t]
                u = u + 1
        group = group_db

    elif (max_time_km < time_slot) and (max_time_db > time_slot):  # choose the clustering such that all orders would be on time
        for b in range(len(group_batch_size_km)):
            u = 0
            c = round(group_batch_cost_km[b] / group_batch_size_km[b], 3)
            for t in range(int(group_batch_size_km[b])):
                group_km.loc[group_batch_path_km[b][t], 'terminal'] = u
                group_km.loc[group_batch_path_km[b][t], 'avg_cost'] = c
                group_km.loc[group_batch_path_km[b][t], 'delivery_time'] = group_batch_time_km[b][t]
                u = u + 1
        group = group_km

    return group

def calc_avg_cost_per_order(batch_size, terminal_fare, time_slot):
##########################################################################################################
# Load data and calculate it's statistics
##########################################################################################################
    # data = pd.read_excel('./data/orders_task 2.xlsx')
    # print('Dataframe dimensions:', data.shape)
    # data.to_pickle('./data/data_task2.pkl')
    data = pd.read_pickle('./data/data_task2.pkl')

    data['first_created_at'] = pd.to_datetime(data['first_created_at'])

    order_time_coding(data, prepending_time)
    grouping_temp = data.groupby(by=['contact_name', 'date', 'order_time_code'])
    groups = [grouping_temp.get_group(x) for x in grouping_temp.groups]
    data['group'] = -1

    for g in range(len(groups)):
        data['group'][groups[g]['order_id']] = g

    group_member_counts = data.value_counts('group')

    data['batch'] = -1
    h = 0

    for i in range(len(groups)):
        print()
        if group_member_counts[i] == 1:
            data.loc[data['group'] == i, 'batch'] = 0
            data.loc[data['group'] == i, 'terminal'] = 0
            a1 = data.loc[data['group'] == i, 'source_latitude']
            a2 = data.loc[data['group'] == i, 'source_longitude']
            b1 = data.loc[data['group'] == i, 'destination_latitude']
            b2 = data.loc[data['group'] == i, 'destination_longitude']
            d = distance.distance((a1.iloc[0], a2.iloc[0]), (b1.iloc[0], b2.iloc[0])).kilometers
            data.loc[data['group'] == i, 'avg_cost'] = calc_batch_cost(d, 1, terminal_fare)
            data.loc[data['group'] == i, 'delivery_time'] = calc_batch_time([d])

        # TO_DO
        elif group_member_counts[i] <= 5:
            if group_member_counts[i] > batch_size:
                temp = []
                temp[:batch_size] = np.zeros(batch_size)
                temp[batch_size:group_member_counts[i]] = np.ones(group_member_counts[i] - batch_size)
                data.loc[data['group'] == i, 'batch'] = temp

                temp = []
                temp[:batch_size] = np.arange(batch_size)
                temp[batch_size:group_member_counts[i]] = np.arange(group_member_counts[i] - batch_size)
                data.loc[data['group'] == i, 'terminal'] = temp

                group_batch_distances, group_batch_size, group_batch_cost, group_batch_time, group_max_dist, group_batch_path, group_batch_dist = calc_dist(
                    data.loc[data['group'] == i], terminal_fare)

                temp = []
                temp[:batch_size] = np.ones(batch_size) * round(
                    calc_batch_cost(group_batch_distances[0], group_batch_size[0], terminal_fare) / group_batch_size[0], 3)
                temp[batch_size:group_member_counts[i]] = np.ones(group_member_counts[i] - batch_size) * round(
                    calc_batch_cost(group_batch_distances[1], group_batch_size[1], terminal_fare) / group_batch_size[1], 3)
                data.loc[data['group'] == i, 'avg_cost'] = temp

                temp = []
                temp[:batch_size] = calc_batch_time(group_batch_dist[0]) / group_batch_size[0]
                if group_batch_size[1] > 1:
                    temp[batch_size:group_member_counts[i]] = calc_batch_time(group_batch_dist[1]) / group_batch_size[1]
                else:
                    temp[batch_size:group_member_counts[i]] = calc_batch_time(group_batch_dist[1][0]) / \
                                                              group_batch_size[1]
                data.loc[data['group'] == i, 'delivery_time'] = temp

            else:
                temp = []
                temp[:group_member_counts[i]] = np.zeros(group_member_counts[i])
                data.loc[data['group'] == i, 'batch'] = temp

                temp = []
                temp[group_member_counts[i]:] = np.arange(group_member_counts[i])
                data.loc[data['group'] == i, 'terminal'] = temp

                group_batch_distances, group_batch_size, group_batch_cost, group_batch_time, group_max_dist, group_batch_path, group_batch_dist = calc_dist(
                    data.loc[data['group'] == i], terminal_fare)

                temp = []
                temp[:group_member_counts[i]] = np.ones(group_member_counts[i]) * round(
                    (calc_batch_cost(group_batch_distances, group_batch_size, terminal_fare)[0] / group_member_counts[i]), 3)
                data.loc[data['group'] == i, 'avg_cost'] = temp

                temp = []
                temp[:group_member_counts[i]] = calc_batch_time(group_batch_dist[0]) / group_member_counts[i]
                data.loc[data['group'] == i, 'delivery_time'] = temp

        elif group_member_counts[i] > 5:
            group = clustering_group(data.loc[data['group'] == i], terminal_fare, batch_size, time_slot)
            data.loc[data['group'] == i, 'batch'] = group['batch']
            data.loc[data['group'] == i, 'terminal'] = group['terminal']
            data.loc[data['group'] == i, 'avg_cost'] = group['avg_cost']
            data.loc[data['group'] == i, 'delivery_time'] = group['delivery_time']
            h = h + 1

    avg_cost_per_order = sum(data['avg_cost'] / data.shape[0])


    res = data['delivery_time'] - (np.ones(data.shape[0]) * time_slot)
    on_time = (res[res <= 5].shape[0] / data.shape[0]) * 100
    more_than_five_min_late = (res[res > 5].shape[0] / data.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / data.shape[0]) * 100

    return round(avg_cost_per_order, 3), round(on_time, 2), round(more_than_five_min_late, 2), round(more_than_ten_min_late, 2)

def main_all_parameter_apace():
    batch_size_options = [3, 4, 5]
    terminal_fare_options = [5, 6, 7]
    time_slot_options = [30, 45, 60]

    titles = ['Average Cost per Order', 'on_time', 'more_than_five_min_late', 'more_than_ten_min_late', 'Order Time Slot', 'Terminal Fare', 'Batch Size']
    t = PrettyTable(titles)
    for bs in range(len(batch_size_options)):
        for tf in range(len(terminal_fare_options)):
            for ts in range(len(time_slot_options)):
                temp = []
                avg_cost_per_order, on_time, more_than_five_min_late, more_than_ten_min_late = calc_avg_cost_per_order(batch_size_options[bs], terminal_fare_options[tf], time_slot_options[ts])
                temp.append(avg_cost_per_order)
                temp.append(on_time)
                temp.append(more_than_five_min_late)
                temp.append(more_than_ten_min_late)
                temp.append(time_slot_options[ts])
                temp.append(terminal_fare_options[tf])
                temp.append(batch_size_options[bs])
                t.add_row(temp)

    print(t)

def main(batch_size = 3, time_slot = 45, terminal_fare = 5):

    titles = ['Average Cost per Order', 'on_time', 'more_than_five_min_late', 'more_than_ten_min_late', 'Order Time Slot', 'Terminal Fare', 'Batch Size']
    t = PrettyTable(titles)
    temp = []
    avg_cost_per_order, on_time, more_than_five_min_late, more_than_ten_min_late = calc_avg_cost_per_order(batch_size, terminal_fare, time_slot)
    temp.append(avg_cost_per_order)
    temp.append(on_time)
    temp.append(more_than_five_min_late)
    temp.append(more_than_ten_min_late)
    temp.append(time_slot)
    temp.append(terminal_fare)
    temp.append(batch_size)
    t.add_row(temp)

    print(t)

if __name__ == '__main__':
    main(batch_size = 3, time_slot = 45, terminal_fare = 5)
    # main_all_parameter_apace()