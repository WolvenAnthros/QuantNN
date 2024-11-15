import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pennylane.numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('task-3-dataset.csv', header=0, quotechar='"', skipinitialspace=True, encoding='utf-8')

texts = df['отзывы'].values
labels = df['разметка'].values
labels = [1 if label == '+' else 0 for label in labels]
# Bag-of-Words Embedding
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Apply PCA to reduce dimensionality to 1D
n_qubits = 4
pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X)


# Haar Wavelet Transform Function
def haar_wavelet_transform(data):
    n = len(data)
    if n == 1:
        return data

    # Ensure the length of the data is even
    if n % 2 != 0:
        data = np.pad(data, (0, 1), mode='constant')
        n += 1

    # Split the data into even and odd indexed elements
    even = data[::2]
    odd = data[1::2]

    # Compute approximation and detail coefficients
    approx = (even + odd) / 2
    detail = (even - odd) / 2

    # Recursively apply the transform to the approximation coefficients
    if len(approx) > 1:
        approx = haar_wavelet_transform(approx)

    return np.concatenate((approx, detail))


# Apply Haar Wavelet Transform to each row in X_pca
X_haar = np.array([haar_wavelet_transform(row) for row in X_pca])

# Normalize the data
X_wavelet = (X_haar - X_haar.mean()) / X_haar.std()

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnode(inputs, weights):  # weights_1_layer, weights_2_layer):
    # print(inputs)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # ZZFeatureMap(inputs=inputs, wires=range(n_qubits))
    # qml.StronglyEntanglingLayers(weights=weights_1_layer, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
    # qml.layer(final_layer, depth=1, wires=[2, 3], weights=weights_2_layer)
    return [qml.expval(qml.PauliZ(wires=n_qubits - 1))]


num_layers = 1
weight_shapes = {
    "weights": (num_layers, n_qubits, 3)
}


def accuracy(params):
    predictions = [qnode(x, params) for x in X_wavelet]  # , params[1]
    predictions = np.array(predictions)
    predictions = (predictions > 0).astype(int)
    f1 = f1_score(labels, predictions)
    return np.mean(predictions == labels), f1

trained_params = np.load('trained_params_synthetic.npy')

print(accuracy(params=trained_params))
print(qml.draw(qnode)([1,1,1,1],trained_params))