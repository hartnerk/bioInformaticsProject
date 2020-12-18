# libraries
import math
import time
import operator
import numpy as np	
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from skbio.stats.distance import DissimilarityMatrix

#READ IN DATA
def main():
	rawData = []
	with open("dataset_for_homework.txt", "r") as source:
	    for line in source:
	        fields = line.rstrip().split('\t')
	        rawData.append(fields)

	start_time = time.clock()

	graphArray=[];
	graphArray=eclideanDistance(rawData)

	plotDistance(rawData, graphArray)

	# K-Means
	# Choose X furtherest appart Sampels as your centroids
	samples=len(rawData[0])-1
	numCenteriods=3
	
	ks=kMeansCentriods(numCenteriods , samples ,graphArray)
	print("Furtherest apart samples and starting centriods are: ")
	print(ks)

	centriodsAndSamplesArray=[]
	distanceWithCentriods=[]
	clusters=[]
	lastClusters=clusters
	centriodsAndSamplesArray=addIntialCentriods(rawData,  ks)
	distanceWithCentriods = eclideanDistance(centriodsAndSamplesArray)
	plotDistance(centriodsAndSamplesArray, distanceWithCentriods)
	clusters=sortClusters(distanceWithCentriods, numCenteriods, samples)

	while lastClusters!=clusters:
		lastClusters=clusters
		centriodsAndSamplesArray=setMeanCentriods(clusters, rawData)
		distanceWithCentriods = eclideanDistance(centriodsAndSamplesArray)
		plotDistance(centriodsAndSamplesArray, distanceWithCentriods)
		clusters=sortClusters(distanceWithCentriods, numCenteriods, samples)
	print("Project K-Means Program")
	print (time.clock() - start_time, "seconds")

	#OPEN SOURCE KMeans Clustering algorithms avaible in Python
	#SKLEARN
	cleanedData=[]
	for x in range(1,len(rawData)):
		tempcleaned=[]
		for y in range(1, len(rawData[x])):
			tempcleaned.append(float(rawData[x][y]))
		cleanedData.append(tempcleaned)

	start_time = time.clock()
	X = np.array(cleanedData)
	kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, verbose=0, random_state=None).fit(X)
	#print (
	kmeans.cluster_centers_
	#)
	print("SKLEARN \"AUTO\"  Algorithim")
	print (time.clock() - start_time, "seconds")

	start_time = time.clock()
	kmeans = KMeans(algorithm='full', n_clusters=2, init='k-means++', n_init=10, max_iter=300, verbose=0, random_state=None).fit(X)
	#print (
	kmeans.cluster_centers_
	#)
	print("SKLEARN \"FULL\"  Algorithim")
	print (time.clock() - start_time, "seconds")

	start_time = time.clock()
	kmeans = KMeans(algorithm='elkan', n_clusters=2, init='k-means++', n_init=10, max_iter=300, verbose=0, random_state=None).fit(X)
	#print (
	kmeans.cluster_centers_
	#)
	print("SKLEARN \"ELKAN\"  Algorithim")
	print (time.clock() - start_time, "seconds")

	start_time = time.clock()
	#SCIPY.org
	whitened = whiten(cleanedData)
	book = np.array((whitened[0],whitened[7]))
	centroid, label = kmeans2(cleanedData, 2, minit='points')
	print(centroid)
	print("SCIPY \"KMEAN2\"  Algorithim")
	print (time.clock() - start_time, "seconds")

def sortClusters(distanceMatrix, numCenteriods, samples):
	finalClusters=[]
	finalDistClusters=[]
	i=samples-1
	count=0

	while i < len(distanceMatrix):
		distClusters=[]
		for x in range(0, numCenteriods):
			distClusters.append(distanceMatrix[x+i])
		i+=samples-count+numCenteriods
		count+=1
		finalDistClusters.append(distClusters)

	for x in range(0, numCenteriods):
		tempClusters=[]
		tempClusters.append("Cluster "+str(x+1))
		finalClusters.append(tempClusters)
	
	for x in range(0, len(finalDistClusters)):
		finalClusters[finalDistClusters[x].index(min(finalDistClusters[x]))].append(x)
	print(finalClusters)
	return finalClusters

def addIntialCentriods(rawData, ks):
	intialCentriods=[]
	for x in ks:
		centroids=[]
		for y in range(0, len(rawData)):
			centroids.append(rawData[y][x])
		intialCentriods.append(centroids)
	return addSamples(rawData, intialCentriods)


def addSamples(rawData, intialCentriods):
	for i in range(0, len(rawData)):
		for x in range(0, len(intialCentriods)):
			if i==0:
				rawData[i].append('Centriod '+str(x+1))
			else:
				rawData[i].append(intialCentriods[x][i])
	return rawData

def setMeanCentriods(centroids, rawData):
	finalCentroids=[]
	for x in range(0, len(centroids)):
		centriod=[]
		centriod.append('Centriod '+str(x+1))
		for z in range(1, len(rawData)):
			tempsum=0
			for y in range(1, len(centroids[x])):
				tempsum+=float(rawData[z][centroids[x][y]+1])
			centriod.append(tempsum/(len(centroids[x])-1))
		finalCentroids.append(centriod)
		rawTempData=[]
		for x in rawData:
			rawTempData.append(x[0:len(x)-1])
		rawData=rawTempData
	return addSamples(rawData, finalCentroids)

def eclideanDistance(rawData):
	# EUCLIDEAN DISTANCE IN D-DIMENSIONS
	distArray=[];
	for i in range(1, len(rawData[0])):
		rowArray=[];
		for s in range(i, len(rawData[0])):
			sumDis=0;
			for z in range(1, len(rawData)):
				sumDis=sumDis+(float(rawData[z][i])-float(rawData[z][s]))**2;
				if(z==len(rawData)-1):
					rowArray.append(math.sqrt(sumDis))
			if(s==len(rawData[0])-1):
				distArray.append(rowArray)

	graphArray=[]
	for x in distArray:
	 	for y in x[1:]:
	 		graphArray.append(y)

	return graphArray


def kMeansCentriods(requestedCentriods, samples, distArray):
	ks=[]
	maxVerts=[]
	while len(ks)<requestedCentriods:
		maxIndex=distArray.index(max(distArray))
		count=0
		arraymaxindex=[]
		for i in range(1, samples):
			if maxIndex<(samples-i):
				arraymaxindex.append(count)
				arraymaxindex.append(samples-1-maxIndex)
				if count not in ks:
					ks.append(count)
				if  (samples-1-maxIndex) not in ks:
					ks.append(samples-1-maxIndex)
				distArray[distArray.index(max(distArray))]=0
				break
			else:
				maxIndex=maxIndex-(samples-i)
				count+=1
		maxVerts.append(arraymaxindex)
	return ks

def plotDistance(rawData, graphArray):
	#PLOT
	axis=[]
	for i in rawData[0][1:]:
	 	axis.append(i)
	dm = DissimilarityMatrix(graphArray, axis)
	fig = dm.plot(cmap='Reds', title='Distance Graph')
	plt.show()

main()