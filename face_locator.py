# USAGE
# python encode_faces.py --dataset dataset/ --encoding encodings.pickle --detection-method hog
#
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
import numpy

from importlib.machinery import SourceFileLoader


### servo initialization stuff.
servo_client = SourceFileLoader("ServoClient", "/home/pi/proj/servo_control/servo_test.py").load_module()
# sc = servo_client.ServoClient()

# servo_X = servo_client.ServoClient(servoConfig_X, 10)
# servo_Y = servo_client.ServoClient(servoConfig_Y, 15)

class Config():
    cascade = 'haarcascade_frontalface_default.xml'
    encodings = 'encodings.pickle'
    servoConfig_X = {
        "pin": 11,
        "Hz": 50
    }
    servoConfig_Y = {
        "pin": 12,
        "Hz": 50
    }
    direction_left = 'LEFT'
    direction_right = 'RIGHT'
    direction_up = 'UP'
    direction_down = 'DOWN'


class Process_Manager():
    def __init__(self, config):
        self.config = config
        self.X_duty = 0
        self.Y_duty = 0
        self.servo_X = servo_client.ServoClient(config.servoConfig_X, self.X_duty)
        self.servo_Y = servo_client.ServoClient(config.servoConfig_Y, self.Y_duty)

    def __determine_which_way_to_turn(self, current_face_location, target_location):
        # (top, right, bottom, left)
        x_diff = current_face_location[1] - current_face_location[3]
        y_diff = current_face_location[2] - current_face_location[0]

        x_direction_turn = None
        y_direction_turn = None

        # left
        if x_diff < int((target_location[3] - current_face_location[3])*2):
            x_direction_turn = self.config.direction_left
        # right
        elif x_diff < int((current_face_location[1] - target_location[1])*2):
            x_direction_turn = self.config.direction_right
        # up
        if y_diff < int((target_location[0] - current_face_location[0])*2):
            y_direction_turn = self.config.direction_up
        # down
        elif y_diff < int((current_face_location[2] - target_location[2])*2):
            y_direction_turn = self.config.direction_down

        return {
            'x': x_direction_turn,
            'y': y_direction_turn
        }

    def __move_servo_in_direction(self, direction):
        new_X_duty = self.X_duty
        if direction['x'] == self.config.direction_left:
            new_X_duty = self.X_duty - 1.0
        elif direction['x'] == self.config.direction_right:
            new_X_duty = self.X_duty + 1.0

        new_Y_duty = self.Y_duty
        if direction['y'] == self.config.direction_down:
            new_Y_duty = self.Y_duty - 1.0
        elif direction['y'] == self.config.direction_up:
            new_Y_duty = self.Y_duty + 1.0

        print("Angle adjustment from (%s, %s) to (%s, %s)" % (self.X_duty, self.Y_duty, new_X_duty, new_Y_duty))
        self.__set_servo_position(new_X_duty, new_Y_duty, 0.5)

    def __set_servo_position(self, x_duty, y_duty, stepSize=1):
        do_x = True
        do_y = True
        if x_duty > 22 or x_duty < 2:
            do_x = False
        if y_duty > 22 or y_duty < 2:
            do_y = False
        if not do_x and not do_y:
            return 1
        if x_duty == self.X_duty and y_duty == self.Y_duty:
            return 0

        # while current_duty_cycle < newDutyCycle:
        max_x = max(self.X_duty, x_duty)
        min_x = min(self.X_duty, x_duty)
        do_flip_x = self.X_duty > x_duty
        x_range = numpy.arange(min_x, max_x, stepSize)
        if do_flip_x:
            x_range = numpy.flip(x_range)

        max_y = max(self.Y_duty, y_duty)
        min_y = min(self.Y_duty, y_duty)
        do_flip_y = self.Y_duty > y_duty
        y_range = numpy.arange(min_y, max_y, stepSize)
        if do_flip_y:
            y_range = numpy.flip(y_range)

        max_x_range = len(x_range)
        max_y_range = len(y_range)
        for i in range(max(max_x_range, max_y_range)):
            if do_x and i < max_x_range:
                self.servo_X._servo.ChangeDutyCycle(
                    x_range[i]
                )
            if do_y and i < max_y_range:
                self.servo_Y._servo.ChangeDutyCycle(
                    y_range[i]
                )
            time.sleep(0.5)

        if do_x:
            self.X_duty = x_duty
        if do_y:
            self.Y_duty = y_duty
        self.servo_Y._servo.ChangeDutyCycle(
                    0
                )
        self.servo_X._servo.ChangeDutyCycle(
                    0
                )
        return 0

    def __auto_adjust_servos(self, face_data, stream):
        position_to_move = self.__determine_which_way_to_turn(
            face_data['boxes'][0],
            stream.get_center_of_frame()
        )

        self.__move_servo_in_direction(
            position_to_move
        )

    def run(self, train=False):
        stream = Stream()
        face_finder = Face_Finder(self.config.encodings, self.config.cascade)
        self.__set_servo_position(2,2,0.5)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame = stream.read_frame_and_format()
            face_data = face_finder.find_face_in_frame(frame)
            stream.draw_debug_face_identification(face_data)

            ## adjust position...
            if key == ord("m") and not train:
                self.__auto_adjust_servos(
                    face_data, stream
                )

            ## train method fork
            new_X_duty = self.X_duty
            new_Y_duty = self.Y_duty
            if key == ord("d") and not train:
                #right
                new_X_duty = self.X_duty + 1.0
            if key == ord("a") and not train:
                #left
                new_X_duty = self.X_duty - 1.0
            if key == ord("w") and not train:
                #up
                new_Y_duty = self.Y_duty + 1.0
            if key == ord("s") and not train:
                #down
                new_Y_duty = self.Y_duty - 1.0

            if new_X_duty != self.X_duty or new_Y_duty != self.Y_duty:
                self.__set_servo_position(new_X_duty, new_Y_duty, 0.5)

            if key == ord("k") and not train:
                #save state
                pass


            if key == ord("p"):
                print("x direction: %s | y direction: %s" % (position_to_move['x'], position_to_move['y']))

        stream.tear_down()
        self.__set_servo_position(2,2,0.5)
        #self.servo_X.setDutyCycle(2, 0.1)
        #self.servo_Y.setDutyCycle(2, 0.1)


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

        self.frame = imutils.resize(self.frame, width=500, height=300) # @todo general note if we're consistently resizing then the size of the frame will always be the same and the bottom two things can be made constant...
        (self.frame_height, self.frame_width) = self.frame.shape[:2] ### @todo make a whole separate frame struct so this can be saved in the same place.

        return self.frame

    def get_center_of_frame(self):
        # (top, right, bottom, left)
        return [
            int(self.frame_height / 2 - 20),  # top
            int(self.frame_width / 2 + 20),   # right
            int(self.frame_height / 2 + 20),  # bottom
            int(self.frame_width / 2 - 20)    # left
        ]

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

    def find_face_in_frame(self, frame):
        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        #   or just calculate it from the top, bottom, left, and right param
        closest_face_box = self.__select_closest_face(self.boxes)

        ### This actually identifies who is in the picture, not whether there is someone in frame. that's already done at this point.
        # compute the facial embeddings for each face bounding box
        encodings = []
        #encodings = face_recognition.face_encodings(rgb, [closest_face_box])

        self.names = ['Unknown'] ### just set this to unknown for now...later this'll be populated with the face identification.
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
        # print('box found: %s %s %s %s' % closest_face_box)
        return {'boxes': [closest_face_box], 'names': self.names}

    def __select_closest_face(self, boxes):
        closest_square = (0,0,0,0)
        for (top, right, bottom, left) in boxes:
            if not closest_square:
                closest_square = (top, right, bottom, left)
            elif (right - left > closest_square[1] - closest_square[3]) or \
                 (top - bottom > closest_square[0] - closest_square[2]):
                closest_square = (top, right, bottom, left)

        return closest_square




proc = Process_Manager(Config())
proc.run()
