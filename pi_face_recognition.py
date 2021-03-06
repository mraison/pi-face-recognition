# USAGE
# python encode_faces.py --dataset dataset/ --encoding encodings.pickle --detection-method hog
#
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

from importlib.machinery import SourceFileLoader

servo_client = SourceFileLoader("ServoClient", "/home/pi/proj/servo_control/servo_test.py").load_module()
# sc = servo_client.ServoClient()
servoConfig_X = {
	"pin":11,
	"Hz": 50
}
servoConfig_Y = {
	"pin":12,
	"Hz": 50
}

X_duty = 2
Y_duty = 2
servo_X = servo_client.ServoClient(servoConfig_X, 10)
servo_Y = servo_client.ServoClient(servoConfig_Y, 15)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0, usePiCamera=True).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	
	# mraison - attempting to flip.
	frame = imutils.rotate(frame, 180)
	
	frame = imutils.resize(frame, width=500)
	(frame_height, frame_width) = frame.shape[:2]
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	# (top, right, bottom, left)
	# mraison - I need to either pull the "center" of the squares from here
	# 	or just calculate it from the top, bottom, left, and right param

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
		
	# Center range of what I'll deam is "ok"
		center_square_ok_range = (
			int (frame_width/2 - 20), #left
			int (frame_width/2 + 20), #right
			int (frame_height/2 - 20), #top
			int (frame_height/2 + 20) #bottom
		)
		cv2.rectangle(
			frame, 
			(center_square_ok_range[0], center_square_ok_range[2]), 
			(center_square_ok_range[1], center_square_ok_range[3]),
			(0, 255, 0),
			2
		)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		if not name == "matthew_raison":
			continue
		
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
			
		# after drawing all that stuff then do the servo adjustment
		# (left, top), (right, bottom)
		# frame_height, frame_width
		center_of_face_square = (
			int(((right - left)/2) + left), # x
			int(((bottom - top)/2) + top)   # y
		)
		
		# mraison - finding center.
		cv2.rectangle(
			frame, 
			(center_of_face_square[0] -1, center_of_face_square[1] -1), 
			(center_of_face_square[0] +1, center_of_face_square[1] +1),
			(0, 255, 0),
			2
		)
		if (
			not (center_square_ok_range[0] <= center_of_face_square[0] and #left
			center_square_ok_range[1] >= center_of_face_square[0] and #right
			center_square_ok_range[2] >= center_of_face_square[1] and #top
			center_square_ok_range[3] <= center_of_face_square[1])    #bottom
		):
			if center_of_face_square[0] > int(frame_width/2):
				new_X_duty = X_duty - 0.10
			elif center_of_face_square[0] < int(frame_width/2):
				new_X_duty = X_duty + 0.10
			
			if center_of_face_square[1] > int(frame_height/2):
				new_Y_duty = Y_duty - 0.10
			elif center_of_face_square[1] < int(frame_height/2):
				new_Y_duty = Y_duty + 0.10
			
			# only apply the shift if it's within frame.
			if new_Y_duty <= 22 and new_Y_duty >= 2:
				Y_duty = new_Y_duty
			
			if new_X_duty <= 22 and new_X_duty >= 2:
				X_duty = new_X_duty
			
			print("Angle adjustment found (%s, %s)" % (X_duty,Y_duty) )
			servo_X.setDutyCycle(X_duty, 0.05)
			#time.sleep(0.5)
			servo_Y.setDutyCycle(Y_duty, 0.05)
			#time.sleep(0.5)

	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
