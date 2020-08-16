import numpy as np 
import cv2
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
	print("Usage: python script.py <file>")
	exit(1)

img = cv2.imread(sys.argv[1],1)
processing_height = 1000

#resize image
img = cv2.resize(img,(processing_height, int(img.shape[0] * processing_height / img.shape[1])))

#convert image to grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blurr image to smooth 
blurr = cv2.GaussianBlur(grey, (15,15), 0)

#finding edges 
edge = cv2.Canny(blurr, 20, 50)   

#thicken
# edge_thick = cv2.dilate(edge, (3,3))
_,edge_thick = cv2.threshold(cv2.GaussianBlur(edge, (3,3), 0), 1, 255, cv2.THRESH_BINARY)

#apadtive threshold and canny gave similar final output 
# threshold = cv2.adaptiveThreshold(blurr ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#find contours in thresholded image and sort them according to decreasing area
contours, _ = cv2.findContours(edge_thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(cv2.convexHull(x)), reverse= True)

contoursImg = cv2.cvtColor(np.zeros(img.shape, np.uint8), cv2.COLOR_BGR2GRAY)

#contour approximation
# TODO: try to identify when the contour looks dubious
found = False
for contour in contours:
	hull = cv2.convexHull(contour)
	approx = cv2.approxPolyDP(hull, 0.05*cv2.arcLength(hull, True), True)

	# print(cv2.contourArea(hull), cv2.contourArea(approx), len(approx))

	if len(approx) == 4:
		found = True
		doc = approx.reshape((4,2))

		#draw contours
		# cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
		# cv2.drawContours(img, [doc], -1, (255, 0, 255), 2)
		# cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
		cv2.drawContours(contoursImg, [contour], -1, (255, 255, 255), 2)

		# print(doc)
		cv2.circle(contoursImg, (doc[0][0], doc[0][1]), 50, (0,0,0), -1)
		cv2.circle(contoursImg, (doc[1][0], doc[1][1]), 50, (0,0,0), -1)
		cv2.circle(contoursImg, (doc[2][0], doc[2][1]), 50, (0,0,0), -1)
		cv2.circle(contoursImg, (doc[3][0], doc[3][1]), 50, (0,0,0), -1)
		# cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
		# cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)
		break

# contours2, _ = cv2.findContours(contoursImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# if len(contours2)

if not found:
	print("Drivers license not found")
	exit()

# startpoint = None
# for i in range(0, len(contour) - 5):
# 	a = contour[i][0]
# 	b = contour[i+5][0]
# 	angle = np.rad2deg(np.arctan2(a[1] - b[1], a[0] - b[0]))
# 	print(angle)
# 	i = i + 4

lines = cv2.HoughLines(contoursImg, 1, np.pi/360, 100)
points = []
if lines is not None:
	lines = lines[:,0]
	# print(lines)

	for line in lines:
		line[0] = line[0] / 100

	rot = False
	if any(lines[:,0] < 0.1) or any(lines[:,0] > 3.04):
		# print('bad')
		rot = True
		# lines = np.zi)
#np.tile([0, np.pi/4], (43, 1)).astype('float32'))

		# lines = np.dstack((lines[:,0], (lines[:,1] + np.pi/4)))[0]
		
		for line in lines:
			line[1] = line[1] + np.pi/4
			if line[1] > np.pi:
				line[1] = line[1] - np.pi
				line[0] = -line[0]
		
		# print(lines)
		# print(lines.dtype)

	retval, bestLabels, centers = cv2.kmeans(lines, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

	# plt.scatter(lines[:,0], lines[:,1])
	# plt.scatter(centers[:,0], centers[:,1], c='red')
	# plt.show()
	# for rho,theta in lines:
		# print(rho, ',', theta, sep="")

	# for rho, theta in centers:
		# print(rho, '\t', theta)

	for i in range(0, len(lines)):
		rho, theta = lines[i]
		rho = rho*100
		if rot:
			theta = theta - np.pi/4
			if theta < 0:
				theta = theta + np.pi
				rho = -rho 
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 2000*(-b))
		y1 = int(y0 + 2000*(a))
		x2 = int(x0 - 2000*(-b))
		y2 = int(y0 - 2000*(a))
		# cv2.line(img,(x1,y1),(x2,y2),(255,255,0),2)

	for i in range(0, len(centers)):
	# for i in range(0, len(lines)):
		rho, theta = centers[i]

		rho = rho*100
		if rot:
			theta = theta - np.pi/4
			if theta < 0:
				theta = theta + np.pi
				rho = -rho 

		# print(rho, theta)
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 2000*(-b))
		y1 = int(y0 + 2000*(a))
		x2 = int(x0 - 2000*(-b))
		y2 = int(y0 - 2000*(a))
		# cv2.line(img,(x1,y1),(x2,y2),(255,255,0),2)

		# Find intersection points with later lines
		for j in range(i+1, len(centers)):
			rhoJ, thetaJ = centers[j]
			rhoJ = rhoJ*100
			if rot:
				thetaJ = thetaJ - np.pi/4
				if thetaJ < 0:
					thetaJ = thetaJ + np.pi
					rhoJ = -rhoJ

			aJ = np.cos(thetaJ)
			bJ = np.sin(thetaJ)

			d = a*bJ - b*aJ
			if d != 0:
				point = [int((bJ*rho-b*rhoJ)/d), int((-aJ*rho+a*rhoJ)/d)]
				# print(point)
				if point[0] > 0 and point[0] < contoursImg.shape[1] and point[1] > 0 and point[1] < contoursImg.shape[0]:
					points.append(point)

if len(points) == 4:
	# print('Found better doc')
	doc = np.array(points)

#create a new array and initialize 
# new_doc = np.zeros((4,2), dtype="float32")

new_doc = cv2.convexHull(doc)[:,0].astype('float32')
new_doc = np.roll(new_doc, -np.argmin(new_doc.sum(axis=1)), axis=0)

# Sum = doc.sum(axis = 1)
# new_doc[0] = doc[np.argmin(Sum)]
# new_doc[2] = doc[np.argmax(Sum)]

# Diff = np.diff(doc, axis=1)
# new_doc[1] = doc[np.argmin(Diff)]
# new_doc[3] = doc[np.argmax(Diff)]

# print(new_doc)

# cv2.drawContours(img, [new_doc.astype(int)], -1, (0, 255, 255), 2)

(tl,tr,br,bl) = new_doc

#find distance between points and get max 
dist1 = np.linalg.norm(br-bl)
dist2 = np.linalg.norm(tr-tl)
dist3 = np.linalg.norm(tr-br)
dist4 = np.linalg.norm(tl-bl)

if dist1 + dist2 < dist3 + dist4:
	tmp = tl
	tl = tr
	tr = br
	br = bl
	bl = tmp
	new_doc = np.array([tl,tr,br,bl])

	dist1 = np.linalg.norm(br-bl)
	dist2 = np.linalg.norm(tr-tl)
	dist3 = np.linalg.norm(tr-br)
	dist4 = np.linalg.norm(tl-bl)

maxLen = max(int(dist1),int(dist2))
maxHeight = max(int(dist3), int(dist4))

dst = np.array([[0,0],[maxLen-1, 0],[maxLen-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")

matrix = cv2.getPerspectiveTransform(new_doc, dst)

extraInMilimeters = 0
extraLen = (maxLen-1)*extraInMilimeters/856
extraHeight = (maxHeight-1)*extraInMilimeters/540
crop2 = np.array([[0-extraLen,0-extraHeight],[maxLen-1+extraLen, 0-extraHeight],[maxLen-1+extraLen, maxHeight-1+extraHeight], [0-extraLen, maxHeight-1+extraHeight]], dtype="float32")
new_doc_with_padding = cv2.perspectiveTransform(crop2.reshape(-1,1,2), np.linalg.inv(matrix))
matrix2 = cv2.getPerspectiveTransform(new_doc_with_padding, dst)

warp = cv2.warpPerspective(img, matrix2, (maxLen, maxHeight))

scanned = cv2.resize(warp,(856,540))

# TODO: really this should just cover the license itself, not the border if extraInMilimeters > 0
result = cv2.cvtColor(scanned, cv2.COLOR_BGR2LAB)
a = np.average(result[:, :, 1])
b = np.average(result[:, :, 2])
result[:, :, 1] = result[:, :, 1] - ((a - 128) * (result[:, :, 0] / 255.0) * 1.1)
result[:, :, 2] = result[:, :, 2] - ((b - 128) * (result[:, :, 0] / 255.0) * 1.1)
scanned = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# cv2.rectangle(scanned, (305,82), (700,108), (0, 0, 255), 2)
# cv2.rectangle(scanned, (305,110), (700,134), (255, 0, 0), 2)

#cv2.imwrite("edge.jpg", edge)
#cv2.imwrite("contour.jpg", img)
cv2.imwrite("results/" + sys.argv[1].split('/')[-1], scanned)

# show all images 
# cv2.imshow("Original",img)
# cv2.imshow("Grey",grey)
# cv2.imshow("Blurr",blurr)
# cv2.imshow("Canny_Edge",edge)
# cv2.imshow("Edge_thick",edge_thick)
# # cv2.imshow("Threshold",threshold)
# cv2.imshow("Contours", contoursImg)
# cv2.imshow("Scanned", scanned)

# cv2.waitKey(0)
# cv2.destroyAllWindows()