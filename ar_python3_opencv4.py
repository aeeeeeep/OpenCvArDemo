import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from objloader_simple import *

referenceImage = cv2.imread('/home/pacaep/Tests/OpenCvArDemo/img/referenceImage.png',0)
plt.imshow(referenceImage, cmap = 'gray')
sourceImage = cv2.imread('/home/pacaep/Tests/OpenCvArDemo/img/sourceImage.png',0)
plt.imshow(sourceImage, cmap='gray')

orb = cv2.ORB_create()

referenceImagePts = orb.detect(referenceImage, None)
sourceImagePts = orb.detect(sourceImage, None)

referenceImagePts, referenceImageDsc = orb.compute(referenceImage, referenceImagePts)
sourceImagePts, sourceImageDsc = orb.compute(sourceImage, sourceImagePts)

referenceImageFeatures = cv2.drawKeypoints(referenceImage, referenceImagePts,
											referenceImage, color = (0,255,0), flags = 0)
sourceImageFeatures = cv2.drawKeypoints(sourceImage, sourceImagePts,
											sourceImage, color = (0,255,0), flags = 0)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(referenceImageFeatures, cmap = 'gray')
plt.title('Reference Image Features')
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(sourceImageFeatures,cmap='gray')
plt.title('Source Image Features')
plt.tight_layout()
plt.show()

MIN_MATCHES = 30
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)
sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)
matches = bf.match(referenceImageDsc, sourceImageDsc)
matches = sorted(matches, key = lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    idxPairs = cv2.drawMatches(referenceImage, referenceImagePts,
                                sourceImage, sourceImagePts, matches[:MIN_MATCHES],0,flags =2)

    plt.figure(figsize=(12,6))
    plt.axis('off')
    plt.imshow(idxPairs, cmap='gray')
    plt.title('Matching between features')
    plt.show()

else:
    print("Not enough matches have been found - %d/%d" %(len(matches), MIN_MATCHES))
    matchesMask = None

if len(matches) > MIN_MATCHES:
    sourcePoints = np.float32([referenceImagePts[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = referenceImage.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, homography)

    sourceImageMarker = cv2.polylines(sourceImage, [np.int32(transformedCorners)], True,
                                      255, 5, cv2.LINE_AA)
    
else:
    print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCHES))
    matchesMask = None

drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None,
                      matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(referenceImage, referenceImagePts, sourceImageMarker,
                         sourceImagePts, matches, None, **drawParameters)

plt.figure(figsize=(12, 6))
plt.imshow(result, cmap='gray')
plt.show()





camera_parameters = np.array([[1108.38916, 0,          513.796472],
                              [0,          1111.41724, 661.637500],
                              [0,          0,          1]])

obj = OBJ('/home/pacaep/Tests/OpenCvArDemo/models/fox.obj', swapyz = True)

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography )
    col_1 = rot_and_transl[:,0]
    col_2 = rot_and_transl[:,1]
    col_3 = rot_and_transl[:,2]

    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c,p)
    rot_1 = np.dot(c/np.linalg.norm(c,2) + d / np.linalg.norm(d,2), 1/math.sqrt(2))
    rot_2 = np.dot(c/np.linalg.norm(c,2) - d / np.linalg.norm(d,2), 1/math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3)*6
    h,w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex -1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w / 2, p[1] + h/2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1,1,3), projection)
        imgpts = np.int32(dst)
        
        cv2.fillConvexPoly(img, imgpts, (80, 217, 81))
    return img

sourcePoints = np.float32([referenceImagePts[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

homography, _ = cv2.findHomography(sourcePoints,destinationPoints, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
h, w = referenceImage.shape
corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
transformedCorners = cv2.perspectiveTransform(corners, homography)
frame = cv2.polylines(sourceImage, [np.int32(transformedCorners)], True, 255,3,cv2.LINE_AA)
projection = projection_matrix(camera_parameters, homography)
frame = render(frame, obj, projection, referenceImage, True)

plt.figure(figsize=(6,12))
plt.imshow(frame, cmap='gray')
plt.show()
