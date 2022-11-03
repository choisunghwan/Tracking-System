from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Trackable:
        def __init__(self, objectID, centroid):
                self.objectID = objectID
                self.centroids = [centroid]
                self.directions = []
                self.counted = False
                
class Tracker:
        def __init__(self, maxDisappeared=50):
                self.nextObjectID = 0
                self.objects = OrderedDict()
                self.disappeared = OrderedDict()
                self.maxDisappeared = maxDisappeared
                self.positions = []

        def register(self, centroid):
                self.objects[self.nextObjectID] = centroid
                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        def deregister(self, objectID):
                del self.objects[objectID]
                del self.disappeared[objectID]
                print('DEREGISTERED : ' + str(objectID))
                
        def update(self, rects):
                if len(rects) == 0:
                        for objectID in list(self.disappeared.keys()):
                                self.disappeared[objectID] += 1
                                if self.disappeared[objectID] > self.maxDisappeared:
                                        self.deregister(objectID)

                        return self.objects

                inputCentroids = np.zeros((len(rects), 2), dtype="int")

                for (i, (startX, startY, endX, endY)) in enumerate(rects):
                        cX = int((startX + endX) / 2.0)
                        cY = int((startY + endY) / 2.0)
                        inputCentroids[i] = (cX, cY)

                if len(self.objects) == 0:
                        for i in range(0, len(inputCentroids)):
                                self.register(inputCentroids[i])
                else:
                        objectIDs = list(self.objects.keys())
                        objectCentroids = list(self.objects.values())

                        D = dist.cdist(np.array(objectCentroids), inputCentroids)

                        rows = D.min(axis=1).argsort()

                        cols = D.argmin(axis=1)[rows]

                        usedRows = set()
                        usedCols = set()

                        for (row, col) in zip(rows, cols):
                                if row in usedRows or col in usedCols:
                                        continue

                                objectID = objectIDs[row]
                                self.objects[objectID] = inputCentroids[col]
                                self.disappeared[objectID] = 0
                                
                                usedRows.add(row)
                                usedCols.add(col)

                        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                        
                        print('#############################################')
                        print('np.array(objectCentroids) : ', np.array(objectCentroids), ' inputCentroids : ', inputCentroids)
                        print('D : ', D, ' rows : ', rows, ' cols : ', cols)
                        print('D.shape[0] : ', D.shape[0], ' D.shape[1] : ', D.shape[1])
                        print('usedRows : ', usedRows, ' usedCols : ', usedCols)
                        print('unusedRows : ', unusedRows, ' unusedCols : ', unusedCols)
                        
                        if D.shape[0] >= D.shape[1]:
                                for row in unusedRows:
                                        objectID = objectIDs[row]
                                        self.disappeared[objectID] += 1

                                        if self.disappeared[objectID] > self.maxDisappeared:
                                                self.deregister(objectID)
                        else:
                                for col in unusedCols:
                                        self.register(inputCentroids[col])

                return self.objects
