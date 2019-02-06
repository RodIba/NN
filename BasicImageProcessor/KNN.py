from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse 

#construct the argument parser and parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type = int, default = 1, help = "# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type = int, default = -1, help = "# of jobs for knn distance")

args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))  # get paths to images

# initialize the image processor, load dataset from disk 
# and reshape the data matrix 

sp = SimplePreprocessor(32,32) #rescale all images to 32X32 pixels
sdl = SimpleDatasetLoader(preprocessors = [sp]) # initialize loader 
(data,labels) = sdl.load(imagePaths, verbose = 500) # load images - returns 2-tuple with images and labels
data.reshape((data.shape[0], 3072)) # flatten images into a 3000 x 3072 numpy array 
# 3072 = 32x32x3
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes/(1024*1000.00)))

# build training and testing splits 
#encode labels as integers
le = LabelEncoder()
lables = le.fit_transform(labels)

#partition the data into training and testing splits  using 75% of 
#the data for training and the remianing 25% for testing 

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state=42)

# X - data points
# Y - labels

print("[INFO] evaluating KNN classifier...")
model = KNeighborsCLassifier(n_neighbors=args["neighbors"], 
	n_jobs = args["jobs"])

model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names = le.classes_))




