import numpy as np
import csv

K = 15 #vehicle, should be 15
T = 15 #times
N = 57 #nodes
sizes = [K, T, N]

capacity = 20

# Read edges data from CSV with specified encoding and convert it to a matrix
with open('task-2-adjacency_matrix.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    edges = []
    for i, row in enumerate(reader):
        if i == 0:
            continue  # Skip the header row
        edges.append([float(value) if value != '-' else 1e6 for value in row[1:]])

# Convert edges list to a numpy float32 matrix
distances = np.float32(edges)

# Read nodes data from CSV with specified encoding and flatten to get the tickets
with open('task-2-nodes.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    
    # next(reader)  # Skip header if there is one
    nodes = [float(row[1]) for row in reader]

# Convert the tickets list to a numpy float32 array
tickets = np.float32(nodes)

assert len(distances) == len(tickets)

sights = []
for i, t in enumerate(tickets):
    if t!=0:
        sights.append(i)
print(sights)

ticket_sum = np.sum(tickets)

# function for calculating the flattened index
def calculateindex(offsets, sizes):
    totys_index = 0
    for offset, size in zip(offsets, sizes):
        totys_index *= size
        totys_index += offset
    return totys_index





# Каждая достопримечательнсть посещена не более одного раза
XX1 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    for t in range(T):
        for s in sights:
            for b1 in range(K):
                for t1 in range(T):
                    if ((b!=b1) | (t!=t1)):
                        XX1[calculateindex([b, t, s], sizes), 
                            calculateindex([b1, t1, s], sizes)] = 1
# автобус начинает движение из вокзала
XX2 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    XX2[calculateindex([b, 0, 0], sizes), calculateindex([b, 0, 0], sizes)] = -1


# автобус заканчивает движение в вокзале
XX3 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    XX3[calculateindex([b, T-1, 0], sizes), calculateindex([b, T-1, 0], sizes)] = -1 


# В каждом узле одновременно не более одного автобуса
XX4 = np.zeros([K*T*N, K*T*N])

for t in range(T):
    for s in range(1, N): # на вокзале могут быть много автобусов одновременно
        for b in range(K):
            for b1 in range(K):
                if b!=b1:
                    XX4[calculateindex([b, t, s], sizes), 
                        calculateindex([b1, t, s], sizes)] = 1     

# не нарушем загрузку автобуса

ys = round(np.ceil(np.log2(capacity))) # число добавочных бинарных переменных на один автобус
XX5 = np.zeros([K*T*N, K*T*N])
XY5 = np.zeros([K*T*N, K*ys])
YY5 = np.zeros([K*ys, K*ys])

for b in range(K):
    for t in range(T):
        for s in range(N):
            for t1 in range(T):
                for s1 in range(N):
                    if ((t!=t1) | (s!=s1)):
                        XX5[calculateindex([b, t, s], sizes), 
                                calculateindex([b, t1, s1], sizes)] = 1
                    else:
                        XX5[calculateindex([b, t, s], sizes), 
                                calculateindex([b, t, s], sizes)] = 1*(1-2*capacity)
                                
for b in range(K):
    for t in range(T):
        for s in range(N):
            for y in range(ys):
                XY5[calculateindex([b, t, s], sizes), 
                        calculateindex([b, y], [K, ys])] = 1 * 2**y
                                        
                                        
for b in range(K):
    for y in range(ys):
        for y1 in range(ys):
            if (y == y1):
                YY5[calculateindex([b, y], [K, ys]), 
                        calculateindex([b, y], [K, ys])] = 1 * 2**y * (2**y - 2*capacity)
            else:
                YY5[calculateindex([b, y], [K, ys]), 
                        calculateindex([b, y1], [K, ys])] = 1 * 2**(y+y1)               


# Все клиенты обслужены
XX6 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    for t in range(T):
        for s in sights:
            for b1 in range(K):
                for t1 in range(T):
                    for s1 in sights:
                        if ((b!=b1) | (t!=t1) | (s!=s1)):
                            XX6[calculateindex([b, t, s], sizes), 
                                calculateindex([b1, t1, s1], sizes)] = 1*tickets[s]*tickets[s1]
                        else:
                            XX6[calculateindex([b, t, s], sizes), 
                                calculateindex([b, t, s], sizes)] = 1*tickets[s]*(tickets[s]-2*ticket_sum)                                                       
                            


# loss
XX7 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    for t in range(T):
        for s in range(N):
            for s1 in range(N):
                XX7[calculateindex([b, t, s], sizes), 
                        calculateindex([b, t, s1], sizes)] = 1*distances[s, s1]
                                            

# автобусы существуют
XX8 = np.zeros([K*T*N, K*T*N])

for b in range(K):
    for t in range(T):
        for s in range(N):
            for s1 in range(N):
                if (s!=s1):
                    XX8[calculateindex([b, t, s], sizes), 
                        calculateindex([b, t, s1], sizes)] = 1
                else:
                    XX8[calculateindex([b, t, s], sizes), 
                        calculateindex([b, t, s], sizes)] = 1 * (-1)
                


# np.save('qubo_parts/qubo_1.npy', np.float32(XX1))
# np.save('qubo_parts/qubo_2.npy', np.float32(XX2))
# np.save('qubo_parts/qubo_3.npy', np.float32(XX3))
# np.save('qubo_parts/qubo_4.npy', np.float32(XX4))
# np.save('qubo_parts/qubo_5.npy', np.float32(XX5))
# np.save('qubo_parts/qubo_6.npy', np.float32(XX6))
# np.save('qubo_parts/qubo_7.npy', np.float32(XX7))
# np.save('qubo_parts/qubo_XY.npy',np.float32(XY5))
# np.save('qubo_parts/qubo_YY.npy',np.float32(YY5))
# np.save('qubo_parts/qubo_8.npy',np.float32(XX8))




# penystiy coefficients
penalty_1 = 1000 #1e1
penalty_2 = 1000 #1e1
penalty_3 = 1000 #1e1
penalty_4 = 100 #1e2
penalty_5 = 100 #1e2
penalty_6 = 10 # 1e3
penalty_7 = 1000 # 1e1   
penalty_8 = 100000

# comprise the whole matrix
qubo = np.block([[penalty_1*XX1+penalty_2*XX2+penalty_3*XX3+penalty_4*XX4+penalty_5*XX5+penalty_6*XX6+penalty_7*XX7, penalty_5*XY5],
    [penalty_5*XY5.T, penalty_5*YY5]        
    ])


np.save('schedule_QUBO.npy', np.float32(qubo), )