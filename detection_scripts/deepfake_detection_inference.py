import xdnn_classification as xdnn
from numpy import genfromtxt
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


sub_folders = [(1, 'original'), (0, 'fake')]

# Import data
# X_train = genfromtxt('data_df_X_train_deepfake_ffhq_finetuned.csv', delimiter=',')
# y_train = pd.read_csv('data_df_y_train_deepfake_ffhq_finetuned.csv', delimiter=',',header=None)
X_test = genfromtxt('deepfakedetection/twitter_data_df_X_test_pretrained_dalle_images.csv', delimiter=',')
y_test = pd.read_csv('deepfakedetection/twitter_data_df_y_test_pretrained_dalle_images.csv', delimiter=',',header=None)

# Define Labels and Images for reference
print(X_test.shape)
print(y_test.shape)

# Ensure X_test and y_test have the same number of samples
if X_test.shape[0] != y_test.shape[0]:
    print(f"slicing X_test and y_test to have the same number of samples")
    min_samples = min(X_test.shape[0], y_test.shape[0])
    X_test = X_test[:min_samples]
    y_test = y_test.iloc[:min_samples]


# pd_y_train_labels = y_train[1]
# pd_y_train_images = y_train[0]

pd_y_test_labels = y_test[1]
pd_y_test_images = y_test[0]

# Convert Pandas to Numpy (Required)

# y_train_labels = pd_y_train_labels.to_numpy()
# y_train_images = pd_y_train_images.to_numpy()

y_test_labels = pd_y_test_labels.to_numpy()
y_test_images = pd_y_test_images.to_numpy()

# Data Input (dict)

Input = {}

# Input['Images'] = y_train_images
# Input['Features'] = X_train
# Input['Labels'] = y_train_labels

# print(y_train_labels)

#########################################################################
# xDNN Training

# Model Definition
model = xdnn.xDNNClassifier()

print(dir(model))

# # xDNN learning
# start = time.time()
# x = model.train(Input)
# end = time.time()

# print ("###################### Model Trained ####################")

# print("Training Time: ",round(end - start,2), "seconds")

# # xDNN parameters (optional for investigation)
# print ("###################### Parameters ####################")

# Prototypes =x['xDNNParms']['Parameters']
# total_prototypes = 0
# print('Number of Training Data Samples:', len(X_train))

# for i in range(len(Prototypes)):
#     class_prototypes = len(Prototypes[i]['Prototype'])
#     total_prototypes = total_prototypes + class_prototypes
#     print("Class", sub_folders[i], ":", class_prototypes)

# print("Total   :", total_prototypes)
# print("Prototypes as % of the Training Data Samples:", total_prototypes/len(X_train)*100, "%")


# print ("###################### Visual Prototypes ####################")

# Prototypes = x['xDNNParms']['Parameters']

# total_prototypes = 0

# for i in range(len(Prototypes)):
#     class_prototypes = len(Prototypes[i]['Prototype'])
#     total_prototypes = total_prototypes + class_prototypes
#     print("Number of prototypes Class", i+1, ":", class_prototypes)
#     print("Prototypes : ", Prototypes[i]['Prototype'])
#     print(" ")


# Save xDNN model (optional)
# model.save_model(x,'xDNN_FFHQ_finetuned_weights')
# Load xDNN model (optional)
x = model.load_model('/scratch/user/hasnat.md.abdullah/VCT_2/fake_detection_models/deepfakedetection/xDNN_FFHQ_finetuned_weights')
# print(model.extract_parameters(x))
print ("###################### Validation ####################")

TestData = {}

TestData ['xDNNParms'] =  x['xDNNParms']
# print(dir(model))
# print(model.PrototypesIdentification)

# Prototypes = model.get_params()['Parameters']
# TestData['xDNNParms'] = model.extract_parameters(model)
TestData ['Images'] = y_test_images 
TestData ['Features'] = X_test
TestData ['Labels'] = y_test_labels

start = time.time()
# xDNN Predict
pred= model.predict(TestData)
end = time.time()
print("Validation Time: ",round(end - start,2), "seconds")


# xDNN Results
print ("###################### Results ####################")

model.results(pred['EstLabs'],y_test_labels)
exit()




#######################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib as mpl

print(type(Prototypes[0]["Noc"]))

import numpy as np

data = Prototypes[0]["Centre"]
labels = np.array([])
i=1
for i in range(len(Prototypes)):
    if i != 0:
        data = np.concatenate((data, Prototypes[i]["Centre"]))
    temp = np.full((Prototypes[i]["Noc"], ), i+1)
    labels = np.concatenate((labels, temp))



x_std = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)

pca_data = pca.fit_transform(x_std)


tsne = TSNE(n_components=2, perplexity = 15)
tsne_data = tsne.fit_transform(x_std)


print(pca_data.shape)
print(tsne_data.shape)

print(len(Prototypes))




rainbow = cm.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels))
colors = ListedColormap(rainbow(norm(labels)))

print(tsne_data[:,0])
print(tsne_data[:,1])
scatter = plt.scatter(tsne_data[:,0], tsne_data[:,1], c=labels, cmap=colors)
plt.legend(handles=scatter.legend_elements()[0], labels=sub_folders)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig('TSNE_ffhq.png')
plt.show()



from scipy.spatial import Voronoi, voronoi_plot_2d



minima = min(labels)
maxima = max(labels)

# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
print(minima, maxima)
print(mapper.to_rgba(sub_folders[0][0]  +1), mapper.to_rgba(sub_folders[1][0]+1))

vor = Voronoi(pca_data)
voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(labels[r]), lw=1)

import matplotlib.lines as mlines
handles = []
for i in range(len(sub_folders)):
    handles.append(mlines.Line2D([], [], color=mapper.to_rgba(sub_folders[i][0]+1),
                          markersize=15, label=sub_folders[i][1]))

plt.legend( loc = "lower left", handles=handles)


plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig('Voronoi_ffhq.png')
plt.show()


print(len(labels))
print(len(pca_data))

print(labels)





