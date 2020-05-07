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


### servo initialization stuff.
# servo_client = SourceFileLoader("ServoClient", "/home/pi/proj/servo_control/servo_test.py").load_module()
# # sc = servo_client.ServoClient()
# servoConfig_X = {
#     "pin": 11,
#     "Hz": 50
# }
# servoConfig_Y = {
#     "pin": 12,
#     "Hz": 50
# }
#
# X_duty = 2
# Y_duty = 2
# servo_X = servo_client.ServoClient(servoConfig_X, 10)
# servo_Y = servo_client.ServoClient(servoConfig_Y, 15)

class Process_Manager():
    def __init__(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--cascade", required=True,
                        help="path to where the face cascade resides")
        ap.add_argument("-e", "--encodings", required=True,
                        help="path to serialized db of facial encodings")
        self.args = vars(ap.parse_args())

    def run(self, debug=True):
        stream = Stream()
        face_finder = Face_Finder(self.args["encodings"], self.args["cascade"])
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_data = stream.read_frame_and_format()
            face_data = face_finder.find_face_in_frame(frame_data['gray'], frame_data['rgb'])
            stream.draw_debug_face_identification(face_data)

        stream.tear_down()


class Stream():
    def __init__(self):
        ### This sets up the video stream and doesn't actually pertain to the face recognition
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0, usePiCamera=True).start()
        # vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0) ### @todo yeah remove this shit.
        # start the FPS counter
        self.fps = FPS().start()

    def tear_down(self):
        ### This tears down the video stream and doesn't actually pertain to the face recognition
        # stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vs.stop()

    def read_frame_and_format(self):  #@todo refactor to return the rgb and grey scale elements...and the actual frame probably.
        ### Read in the frame and format a bit. Technically this doesn't pertain to the actually face recognition but rather just stream sampling.
        # tldr you can remove this whole function from the face_finder.
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        self.frame = self.vs.read()

        # attempting to flip.
        self.frame = imutils.rotate(self.frame, 180)

        self.frame = imutils.resize(self.frame, width=500) # @todo general note if we're consistently resizing then the size of the frame will always be the same and the bottom two things can be made constant...
        (self.frame_height, self.frame_width) = self.frame.shape[:2] ### @todo make a whole separate frame struct so this can be saved in the same place.

        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return {
            'frame': self.frame,
            'gray': self.gray,
            'rgb': self.rgb
        }

    def draw_debug_face_identification(self, input_from_face_finder):
        ### just by the way this is relying on the assumption that we'll only have one face present.
        ### not technically part of the face recognition
        # Center range of what I'll deam is "ok"
        center_square_ok_range = (
            int(self.frame_width / 2 - 20),  # left
            int(self.frame_width / 2 + 20),  # right
            int(self.frame_height / 2 - 20),  # top
            int(self.frame_height / 2 + 20)  # bottom
        )
        cv2.rectangle(
            self.frame,
            (center_square_ok_range[0], center_square_ok_range[2]),
            (center_square_ok_range[1], center_square_ok_range[3]),
            (0, 255, 0),
            2
        )

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(input_from_face_finder['boxes'], input_from_face_finder['names']):
            if not name == "matthew_raison":
                continue

            # draw the predicted face name on the image
            cv2.rectangle(self.frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)

            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(self.frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

            # after drawing all that stuff then do the servo adjustment
            # (left, top), (right, bottom)
            # frame_height, frame_width
            center_of_face_square = (
                int(((right - left) / 2) + left),  # x
                int(((bottom - top) / 2) + top)  # y
            )

            # mraison - finding center.
            cv2.rectangle(
                self.frame,
                (center_of_face_square[0] - 1, center_of_face_square[1] - 1),
                (center_of_face_square[0] + 1, center_of_face_square[1] + 1),
                (0, 255, 0),
                2
            )

        ### not technically part of the face recognition
        # display the image to our screen
        cv2.imshow("Frame", self.frame)

        # update the FPS counter
        self.fps.update()


class Face_Finder():
    def __init__(self, encodings, cascade):
        ### This loads the faces for face recognition, so this can stay.
        # load the known faces and embeddings along with OpenCV's Haar
        # cascade for face detection
        print("[INFO] loading encodings + face detector...")
        self.data = pickle.loads(open(encodings, "rb").read())  # @todo replace with param
        self.detector = cv2.CascadeClassifier(cascade)  # @todo replace with param

    def find_face_in_frame(self, gray, rgb):
        ### actually does face detection stuff.
        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        self.boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects] ### This is the main thing we care about for the return. @todo are there raiting associated with these? I'd rather only track one.
        # (top, right, bottom, left)
        # mraison - I need to either pull the "center" of the squares from here
        # 	or just calculate it from the top, bottom, left, and right param

        ### This actually identifies who is in the picture, not whether there is someone in frame. that's already done at this point.
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, self.boxes)
        self.names = [] #["Unknown"] ### just set this to unknown for now...later this'll be populated with the face identification.
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
                                                     encoding) ### @todo edit this module so that we only check for one face at a time.
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
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            self.names.append(name) ### This is the main thing we care about for the return.
        return {'boxes': self.boxes, 'names': self.names}


proc = Process_Manager()
proc.run()
