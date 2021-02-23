from __future__ import print_function


import numpy as np
import cv2
import time
import sys
import math


count = 0


def draw_flow(img, flow, step=16):
	
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (x2, y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
	return vis


def draw_hsv(flow):
	(h, w) = flow.shape[:2]
	(fx, fy) = (flow[:, :, 0], flow[:, :, 1])
	ang = np.arctan2(fy, fx) + np.pi
	v = np.sqrt(fx * fx + fy * fy)
	hsv = np.zeros((h, w, 3), np.uint8)
	hsv[..., 0] = ang * (180 / np.pi / 2)
	hsv[..., 1] = 0xFF
	hsv[..., 2] = np.minimum(v * 4, 0xFF)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	cv2.imshow('hsv', bgr)
	return bgr


def warp_flow(img, flow):
	(h, w) = flow.shape[:2]
	flow = -flow
	flow[:, :, 0] += np.arange(w)
	flow[:, :, 1] += np.arange(h)[:, np.newaxis]
	res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
	return res

if __name__ == '__main__':
	prev = cv2.imread("pic1.png")

	prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
	show_hsv = True
	show_glitch = False
	cur_glitch = prev.copy()
	color = (0, 0, 255)
	colorTwo = (255, 23, 124)
	heading = 0
	while True:
		try:
			image = cv2.imread("pic.png")
			grays = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			blurred = cv2.GaussianBlur(grays, (5, 5), 0)
			wide = cv2.Canny(blurred, 50, 150)
			vis = image.copy()
			flow = cv2.calcOpticalFlowFarneback(prevgray,wide,None,0.5,5,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
			prevgray = wide
			cv2.imshow('flow', draw_flow(wide, flow))
			he, we, ce = image.shape
			midPointX = we / 2
			midPointY = he / 2
			weArr = []
			h = 1
			while h < 8:
				jh = (we / 7) * h
				weArr.append(int(round(jh)))
				h += 1
			print(we)
			print(weArr)
			midArr = [0, 0, 0, 0, 0, 0, 0]

			weArrUp = (we / 2) + 50
			weArrDown = (we / 2) - 50

			heArrUp = (he / 2) + 50
			heArrDown = (he / 2) - 50
			#cv2.circle(vis, (int(weArrUp), int(heArrUp - 100)), 15, color, 2)
			#cv2.circle(vis, (int(weArrDown), int(heArrDown)), 15, colorTwo, 2)
			if show_hsv:
				gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
				thresh = cv2.threshold(gray1, 25, 0xFF,
									   cv2.THRESH_BINARY)[1]
				thresh = cv2.dilate(thresh, None, iterations=2)
				cv2.imshow('thresh', thresh)
				cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				leftVal = 0
				rightVal = 0
				upVal = 0
				downVal = 0
				centerValX = 0
				centerValY = 0
				for c in cnts:

					(x, y, w, h) = cv2.boundingRect(c)
					if w > 100 and h > 100 and w < 900 and h < 680:
						xDim = x + (w / 2)
						yDim = y + (h / 2)
						j = 0
						while j < len(weArr):
							if (weArr[j] - (we / 7)) <= xDim <= weArr[j]:
								midArr[j] = (midArr[j] + 1)
							j += 1

						if xDim < weArrDown:
							leftVal += 1
						elif xDim > weArrUp:
							rightVal += 1
						else:
							centerValX += 1

						if yDim < heArrDown:
							upVal += 1
						elif yDim > heArrUp:
							downVal += 1
						else:
							centerValY += 1

						cv2.circle(vis, (int(xDim), int(yDim)), 15, color, 2)
						cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)
						cv2.putText(vis,str(time.time()),(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0xFF),1)
				#print(midArr)
				if leftVal > rightVal:
					print("Left")
				elif rightVal > leftVal:
					print("Right")
				else:
					print("Center")
				cv2.imshow('Image', vis)
			if show_glitch:
				cur_glitch = warp_flow(cur_glitch, flow)
				cv2.imshow('glitch', cur_glitch)
			ch = 0xFF & cv2.waitKey(5)
			if ch == 27:
				break
			if ch == ord('1'):
				show_hsv = not show_hsv
				print ('HSV flow visualization is', ['off', 'on'][show_hsv])
			if ch == ord('2'):
				show_glitch = not show_glitch
				if show_glitch:
					cur_glitch = wide.copy()
				print ('glitch is', ['off', 'on'][show_glitch])
		except:
			print("Error")
			ch = 0xFF & cv2.waitKey(5)
			if ch == 27:
				with open(file2, 'w') as filetowrite:
					filetowrite.write("land")
				break
	cv2.destroyAllWindows()