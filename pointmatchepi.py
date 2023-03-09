#%%
# importing ...
print('loading pre-processed data from 20220318 and calibrating cameras ...')
cvri=True

import os
import sys
from cv2 import convertFp16
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
from PIL import Image
import optimap as om
import cv2
if(cvri):
    import monochrome as mc

# see also:
#https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
#https://www.geeksforgeeks.org/python-opencv-epipolar-geometry/
#https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/
#https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv
#maybe: https://ros-developer.com/2018/12/21/computing-fundamental-matrix-and-drawing-epipolar-lines-for-stereo-vision-cameras-in-opencv/



path_data = "../Data 20220318/";
path_calibration = 'calibration_parameters/';



print('loading video and calibration data ...')
filename_video1 = path_data + "video1.npy";
filename_video2 = path_data + "video2.npy";
print('loading video1: ' + filename_video1)
video1 = np.load(filename_video1)
print('loading video2: ' + filename_video2)
video2 = np.load(filename_video2)
filename_video1_warped = path_data + "video1_warped.npy";
filename_video2_warped = path_data + "video2_warped.npy";
print('loading video1_warped: ' + filename_video1_warped)
video1_warped = np.load(filename_video1_warped)
print('loading video2_warped: ' + filename_video2_warped)
video2_warped = np.load(filename_video2_warped)
print(video1_warped.shape)
video1_warped = video1_warped[:-60,:,:]
video2_warped = video2_warped[:-60,:,:]
print(video1_warped.shape)
filename_calibration1 = path_data + "calibration1.npy";
filename_calibration2 = path_data + "calibration2.npy";
calibration1 = np.load(filename_calibration1)
calibration2 = np.load(filename_calibration2)
print(calibration1.shape)
print('rotating all videos ...')
video1 = om.video.rotate_left(video1)
video2 = om.video.rotate_left(video2)
video1_warped = om.video.rotate_left(video1_warped)
video2_warped = om.video.rotate_left(video2_warped)
calibration1 = om.video.rotate_left(calibration1)
calibration2 = om.video.rotate_left(calibration2)
im1 = video1_warped[0,:,:]
im2 = video2_warped[0,:,:]
print('displaying raw video images ...')
figure1, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=1)
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=1)
#figure1.show()
print('displaying calibration ...')
figure2, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(calibration1[0],cmap='gray', vmin=0, vmax=4000)
axarr[1].imshow(calibration2[0],cmap='gray', vmin=0, vmax=4000)
#figure2.show()
figure3, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(calibration1[4],cmap='gray', vmin=0, vmax=4000)
axarr[1].imshow(calibration2[4],cmap='gray', vmin=0, vmax=4000)
#figure3.show()







# parameters
chessboardSize = (7, 11); 
print('chessboardSize = (' + str(chessboardSize[0]) + ',' +str(chessboardSize[1]) + ')')
d = 2.5 # distance between corners / grid spacing = 2.5 mm
print('grid distance: ' + str(d) + 'mm')
frameSize = (320, 320); # is this the actual image or sensor?
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria (?)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)
objp = objp * d
print(objp)
# arrays for storing object points and image points from all the images
objPoints = [] # 3D points in real world space
imgPoints1 = [] # 2D points in image plane in camera 1
imgPoints2 = [] # 2D points in image plane in camera 2










print('performing intrinsic camera calibration ...')
print('number of calibration images available for left camera: ' + str(calibration1.shape[0]))
print('number of calibration images available for right camera: ' + str(calibration2.shape[0]))

n = 4;
print('detecting chessboard corners in ' + str(n) + ' images ...')

for i in range(n):
    print('    detecting chessboard corners in image pair ' + str(i))
    im1 = calibration1[n,:,:]
    im2 = calibration2[n,:,:]
    print("    normalizing calibration images")
    im1 = (im1-np.min(im1))/(np.max(im1)-np.min(im1))*255
    im1 = im1.astype(dtype=np.uint8)

    im2 = (im2-np.min(im2))/(np.max(im2)-np.min(im2))*255
    im2 = im2.astype(dtype=np.uint8)

    im_left = Image.fromarray(im1)
    im_left.save("image_left.png")
    im_right = Image.fromarray(im2)
    im_right.save("image_right.png")

    ImgL = cv2.imread("image_left.png")
    ImgR = cv2.imread("image_right.png")

    print('    camera 1 ...')
    ret1, corners1 = cv2.findChessboardCorners(ImgL, chessboardSize, None)
    #cv2.drawChessboardCorners(ImgLrgb.copy(), (7,11), corners1, ret1)
    print('    camera 2 ...')
    ret2, corners2 = cv2.findChessboardCorners(ImgR, chessboardSize, None)
    #cv2.drawChessboardCorners(ImgLrgb.copy(), (7,11), corners1, ret1)
    
    if ret1 and ret2 == True:
        print('    adding point data ...')
        objPoints.append(objp)
        #corners1 = cv2.cornerSubPix(ImgL, corners1, (11,11), (-1,-1), criteria)
        imgPoints1.append(corners1)
        #corners2 = cv2.cornerSubPix(ImgR, corners2, (11,11), (-1,-1), criteria)
        imgPoints2.append(corners2)



figure_calibration1, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=255)
axarr[0].plot(corners1[:,0,0],corners1[:,0,1],'r.')
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=255)
axarr[1].plot(corners2[:,0,0],corners2[:,0,1],'r.')
#axarr[i].set_xlim([0, Nt])
#figure_calibration1.show()

figure_calibration2, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=140)
axarr[0].plot(corners1[:,0,0],corners1[:,0,1],'r.', markersize=10)
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=140)
axarr[1].plot(corners2[:,0,0],corners2[:,0,1],'r.', markersize=10)
axarr[0].set_xlim([100, 200])
axarr[0].set_ylim([200, 100])
axarr[1].set_xlim([100, 200])
axarr[1].set_ylim([200, 100])
#figure_calibration2.show()




# return values for the below functions:
# rmse1 = projection error camera 1
# rmse2 = projection error camera 2
# dc1 = camera 1 distortion coefficients
# dc2 = camera 2 distortion coefficients
# CM1 = camera matrix 1
# CM2 = camera matrix 2
#r1 = rotation vector per frame camera 1
#t1 = translation vector per frame camera 1 
#r2 = rotation vector per frame camera 2
#t2 = translation vector per frame camera 2 

print('perform (intrinsic) camera calibration for each camera individually with cv.calibrateCamera() ...')
rmse1, CM1, dc1, r1, t1 = cv2.calibrateCamera(objPoints, imgPoints1, frameSize, None, None)
print('per pixel projection error camera 1 (rmse): ' + str(rmse1))
rmse2, CM2, dc2, r2, t2 = cv2.calibrateCamera(objPoints, imgPoints2, frameSize, None, None)
print('per pixel projection error camera 2 (rmse): ' + str(rmse2))


mean_reproj_error1 = 0
for i in range(len(objPoints)):
    imgPoints1P, _ = cv2.projectPoints(objPoints[i], r1[i], t1[i], CM1, dc1)
    error1 = cv2.norm(imgPoints1[i], imgPoints1P, cv2.NORM_L2)/len(imgPoints1P)
    mean_reproj_error1 += error1
print("mean reprojection error camera 1: " + str(mean_reproj_error1))

mean_reproj_error2 = 0
for i in range(len(objPoints)):
    imgPoints2P, _ = cv2.projectPoints(objPoints[i], r2[i], t2[i], CM2, dc2)
    error2 = cv2.norm(imgPoints2[i], imgPoints2P, cv2.NORM_L2)/len(imgPoints2P)
    mean_reproj_error2 += error2
print("mean reprojection error camera 2: " + str(mean_reproj_error2))

save_intrinsic = False;
load_intrinsic = False;

if(save_intrinsic):
    np.save(path_calibration + 'cameraMatrix1.npy', CM1)
    np.save(path_calibration + 'distortion_coefficients1.npy', dc1)
    np.save(path_calibration + 'cameraMatrix2.npy', CM2)
    np.save(path_calibration + 'distortion_coefficients2.npy', dc2)
if(load_intrinsic):
    print('loading intrinsic camera calibrations ...')
    CM1 = np.load(path_calibration + 'cameraMatrix1.npy')
    CM2 = np.load(path_calibration + 'cameraMatrix2.npy')
    dc1 = np.load(path_calibration + 'distortion_coefficients1.npy')
    dc2 = np.load(path_calibration + 'distortion_coefficients2.npy')

height, width, channels = 440, 320, 1
#cameraMatrix1new, roi1 = cv2.getOptimalNewCameraMatrix(cameraMatrix1, dist1, (width, height), 1, (width, height))
#cameraMatrix2new, roi2 = cv2.getOptimalNewCameraMatrix(cameraMatrix2, dist2, (width, height), 1, (width, height))

print('perform stereo calibration with cv.stereoCalibrate() ...')
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC # this fixes the intrinsic camera matrix part, so that only rot, transl, Emat and Fmat are calculated
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# R = rotation matrix
# T = translation matrix
# E = essential matrix
# F = fundamental matrix

rmseStereo, CM1new, dist1, CM2new, dist2, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPoints1, imgPoints2, CM1, dc1, CM2, dc2, (width, height), criteria = criteria, flags = flags)
print('per pixel projection error stereo (rmse): ' + str(rmseStereo))
#F = cv2.findFundamentalMat(imgPoints1, imgPoints2, cv2.FM_8POINT, 'ransacReprojThreshold') # is this even necessary?

save_extrinsic = True;
load_extrinsic = False;

if(save_extrinsic):
    np.save(path_calibration + 'R.npy', R)
    np.save(path_calibration + 'T.npy', T)
    np.save(path_calibration + 'E.npy', E)
    np.save(path_calibration + 'F.npy', F)
if(load_extrinsic):
    print('loading intrinsic camera calibrations ...')
    R = np.save(path_calibration + 'R.npy', R)
    T = np.save(path_calibration + 'T.npy', T)
    E = np.save(path_calibration + 'E.npy', E)
    F = np.save(path_calibration + 'F.npy', F)



# %%

print('compute and display epipolar lines')

# https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv


colors = [  [0.5,0,1],
            [1,0,1], 
            [1,0,0], 
            [1,0.5,0], 
            [1,1,0], 
            [0,1,0], 
            [0,1,1], 
            [0, 0, 1]]


p1 = [  [108, 65], 
        [230, 74], 
        [123, 110], 
        [223, 140], 
        [150, 160], 
        [98, 186], 
        [147, 212], 
        [180, 244]] # points / pixels in left camera (camera 1)

p1 = np.array(p1)

p2 = [  [104, 33], 
        [83, 53], 
        [110, 70], 
        [119, 114], 
        [220, 140], 
        [130, 170], 
        [151, 215], 
        [130, 260]]

p2 = np.array(p2)

Np = p1.shape[0]
print('number of pixels in camera image 1: ' + str(Np))
#l2 = 

epilines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F)
epilines1 = epilines1.reshape(-1, 3)
epilines1b = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 1, F)
epilines1b = epilines1b.reshape(-1, 3)

epilines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F) # this is the correct one
epilines2 = epilines2.reshape(-1, 3)
epilines2b = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 2, F) # this is false one
epilines2b = epilines2b.reshape(-1, 3)
print('epilines:')
print(epilines2.shape)
print(epilines2)

# https://hasper.info/opencv-draw-epipolar-lines/
# https://stackoverflow.com/questions/45044784/recognizing-vector-equation-of-line-segment-python

im1 = video1_warped[0,:,:]
im2 = video2_warped[0,:,:]





figureEpi12, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=1)
axarr[0].set_xlim([0, 319])
axarr[0].set_ylim([319, 0])
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=1)
axarr[1].set_xlim([0, 319])
axarr[1].set_ylim([319, 0])
for i in range(0, Np):
    axarr[0].plot(p1[i, 0], p1[i, 1], '.', markersize=12, color=colors[i])
print('draw epilines ...')
for i in range(0,Np):
    x0, y0 = map(int, [0, -epilines2[i][2]/epilines2[i][1]])
    x1, y1 = map(int, [320, -(epilines2[i][2]+epilines2[i][0]*320)/epilines2[i][1]])
    #print('x0: ' + str(x0) + ', x1: ' + str(x1) + ', y0: ' + str(y0) + ', y1: ' + str(y1))
    axarr[1].plot([x0, x1],[y0, y1],color=colors[i], alpha=1.0)
    #x0, y0 = map(int, [0, -epilines2b[i][2]/epilines2b[i][1]])
    #x1, y1 = map(int, [320, -(epilines2b[i][2]+epilines2b[i][0]*320)/epilines2b[i][1]])
    #axarr[1].plot([x0, x1],[y0, y1],color=colors1[i], alpha=0.3)
#figureEpi12.show()


figureEpi21, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=1)
axarr[0].set_xlim([0, 319])
axarr[0].set_ylim([319, 0])
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=1)
axarr[1].set_xlim([0, 319])
axarr[1].set_ylim([319, 0])
for i in range(0, Np):
    axarr[1].plot(p2[i, 0], p2[i, 1], '.', markersize=12, color=colors[i])
print('draw epilines ...')
for i in range(0,Np):
    x0, y0 = map(int, [0, -epilines1[i][2]/epilines1[i][1]])
    x1, y1 = map(int, [320, -(epilines1[i][2]+epilines1[i][0]*320)/epilines1[i][1]])
    axarr[0].plot([x0, x1],[y0, y1],color=colors[i], alpha=1.0)

    #x0, y0 = map(int, [0, -epilines1b[i][2]/epilines1b[i][1]])
    #x1, y1 = map(int, [320, -(epilines1b[i][2]+epilines1b[i][0]*320)/epilines1b[i][1]])
    #axarr[0].plot([x0, x1],[y0, y1],color=colors[i], alpha=0.3)
#axarr[1].set_xlim([0, 320])
#axarr[1].set_ylim([0, 320])
figureEpi21.show()




# %%
print('perform stereo triangulation ...')

def convert_im_to_cv(im):
    print('necessary conversion ...')
    im = (im - np.min(im))/ (np.max(im)-np.min(im))*255
    im = im.astype(dtype=np.uint8)
    im = Image.fromarray(im)
    im.save("temp.png")
    im = cv2.imread("temp.png")
    return im


print('find corresponding points ...')
#hard-coded corresponding points:
import random 

#cp1 = [[145, 155], [170, 200], [120, 100], [212, 167], [233, 150], [100, 166], [160, 118], [170, 170]]
#cp2 = [[100, 152], [135, 204], [80, 100], [175, 170], [200, 150], [68, 166], [120, 115], [130, 170]]
#cp1 = np.array(cp1)
#cp2 = np.array(cp2)


im1 = video1_warped[0];
im2 = video2_warped[0];

im1 = convert_im_to_cv(im1);
im2 = convert_im_to_cv(im2);

print('using sift to detect keypoints ...')
# Detect the SIFT key points and compute the descriptors for the two images
sift = cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

# create BFMatcher object
bf = cv2.BFMatcher()
# Match descriptors.
#matches = bf.match(descriptors1, descriptors2)
matches = bf.knnMatch(descriptors1,descriptors2,k=2) #matches = bf.match(des1,des2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
good = []
for m,n in matches:
    if True :
        good.append([m])

# Draw first 10 matches.
#img3 = cv2.drawMatches(A,kp1,B,kp2,matches[:40],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.drawMatchesKnn(im1, keypoints1, im2, keypoints2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#print_img(img3)


im1_points = [keypoints1[m[0].queryIdx].pt for m in good]
im2_points = [keypoints2[m[0].trainIdx].pt for m in good]
cp1 = np.array(im1_points)
cp2 = np.array(im2_points)
cp1.shape




# Create FLANN matcher object
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# Apply ratio test
goodMatches = []
ptsLeft = []
ptsRight = []
   
#for m, n in matches: 
#    if m.distance < 0.8 * n.distance:
#        goodMatches.append([m])
#        ptsLeft.append(keyPoints1[m.trainIdx].pt)
#        ptsRight.append(keyPoints2[n.trainIdx].pt)



Np = cp1.shape[0]
colors = []
colors.append([1, 0, 0]) #red
colors.append([0, 1, 0]) #blue
colors.append([0, 0, 1]) #green
colors.append([1, 1, 0])
colors.append([1, 0, 1])
colors.append([0, 1, 1])
colors.append([1, 0.5, 0])
colors.append([1, 0, 0.5])
colors.append([0.5, 1, 0])
colors.append([0, 1, 0.5])
colors.append([0.5, 0, 1])
colors.append([0.0, 0.5, 1])

for i in range(0,Np):
    colors.append([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])


im1 = video1_warped[0,:,:]
im2 = video2_warped[0,:,:]


Np = 50
figureCP, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(im1,cmap='gray', vmin=0, vmax=1)
#axarr[0].scatter(cp1[:,0], cp1[:,1], color='red')
for i in range(0, Np):
    axarr[0].plot(cp1[i, 0], cp1[i, 1], '.', markersize=12, color=colors[i])
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=1)
for i in range(0, Np):
    axarr[1].plot(cp2[i, 0], cp2[i, 1], '.', markersize=12, color=colors[i])
#axarr[i].set_xlim([0, Nt])
#figureCP.show()


print('obtaining projection matrices ...')

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1) #RT matrix for C1 is identity.
P1 = CM1 @ RT1 #projection matrix for C1

RT2 = np.concatenate([R, T], axis = -1) #RT matrix for C2 is the R and T obtained from stereo calibration.
P2 = CM2 @ RT2 #projection matrix for C2

print('performing direct linear transform (DLT)')

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


p3ds = []
for cp1i, cp2i in zip(cp1, cp2):
    _p3d = DLT(P1, P2, cp1i, cp2i)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)
print('reconstructed 3d points')
#print(p3ds)

from mpl_toolkits.mplot3d import Axes3D
 
figure3d = plt.figure()
ax = figure3d.add_subplot(111, projection='3d')
#ax.set_xlim3d(-15, 5)
#ax.set_ylim3d(-10, 10)
#ax.set_zlim3d(10, 30)
for i in range(0,50):
    ax.plot(p3ds[i,0], p3ds[i,1], p3ds[i,2], marker = '.', color='red')
#connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
#for _c in connections:
    #print(p3ds[_c[0]])
    #print(p3ds[_c[1]])
    #ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], marker = '.', color = 'red')
#plt.show()
#ax.axis('equal')
#figure3d.show()

print('computing corresponding coordinate along epipolar line ...')
dx = float(dxi[min_idx])
p2bx = x0 + dx
b = (y1-y0)/x1
p2by = y0 + b*p2bx
p2x0 = int(p2bx-wr)
p2x1 = int(p2bx+wr)
p2y0 = int(p2by-wr)
p2y1 = int(p2by+wr)
p2x, p2y = create_rectangle(p2x0, p2x1, p2y0, p2y1) 
print('point 2: (' + str(p2bx) +',' + str(p2by) + ')')



figureEpi12closeup1, axarr = plt.subplots(1,2) # vertical, horizontal
print('plot point p1 in im1 (camera 1)')
axarr[0].imshow(video1[0,:,:],cmap='gray', vmin=0, vmax=1)
axarr[0].set_xlim([p1[i,0]-Wr, p1[i,0]+Wr])
axarr[0].set_ylim([p1[i,1]+Wr, p1[i,1]-Wr])
axarr[0].plot(p1[i, 0], p1[i, 1], '.', markersize=12, color=colors[i])
axarr[0].plot(p1x, p1y, color=colors[i])
axarr[1].imshow(video2[0],cmap='gray', vmin=0, vmax=1)
axarr[1].set_xlim([p2bx-Wr, p2bx+Wr])
axarr[1].set_ylim([p2by+Wr, p2by-Wr])
axarr[1].plot([x0, x1],[y0, y1],color=colors[i], alpha=1.0) # epiline
axarr[1].plot(p2bx, p2by, '.', color=colors[i], markersize=12) # this is the point in the second image
axarr[1].plot(p2x, p2y, color=colors[i])
figureEpi12closeup1.tight_layout(pad=0.9, w_pad=0.4, h_pad=0.5)
figureEpi12closeup1.show()
[2:05 PM]
figureEpi12closeup2, axarr = plt.subplots(1,2) # vertical, horizontal
axarr[0].imshow(video1_contrast[0],cmap='gray', vmin=0, vmax=1)
axarr[0].set_xlim([p1[i,0]-Wr, p1[i,0]+Wr])
axarr[0].set_ylim([p1[i,1]+Wr, p1[i,1]-Wr])
axarr[0].plot(p1[i, 0], p1[i, 1], '.', markersize=12, color=colors[i])
axarr[0].plot(p1x, p1y, color=colors[i])
axarr[1].imshow(im2,cmap='gray', vmin=0, vmax=1)
axarr[1].set_xlim([p2bx-Wr, p2bx+Wr])
axarr[1].set_ylim([p2by+Wr, p2by-Wr])
axarr[1].plot([x0, x1],[y0, y1],color=colors[i], alpha=1.0) # epiline
axarr[1].plot(p2bx, p2by, '.', color=colors[i], markersize=12) # this is the point in the second image
axarr[1].plot(p2x, p2y, color=colors[i])
figureEpi12closeup2.tight_layout(pad=0.9, w_pad=0.4, h_pad=0.5)
figureEpi12closeup2.show()



figureEpi12closeup3, axarr = plt.subplots(1,2) # vertical, horizontal
im1p = im1[p1y0:p1y1+1,p1x0:p1x1+1] # patch from the left camera image (camera 1)
im2p = im2[p2y0:p2y1+1,p2x0:p2x1+1] # patch from the left camera image (camera 1)
axarr[0].imshow(im1p, cmap='gray', vmin=0, vmax=1)
axarr[0].plot(wr, wr, '.', color=colors[i], markersize=12)
axarr[1].imshow(im2p, cmap='gray', vmin=0, vmax=1)
axarr[1].plot(wr, wr, '.', color=colors[i], markersize=12)
#axarr[1,1].set_xlim([0, 319])
#axarr[1,1].set_ylim([319, 0])
figureEpi12closeup3.tight_layout(pad=0.9, w_pad=0.4, h_pad=0.5)
figureEpi12closeup3.show()