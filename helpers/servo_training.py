import json
import os


class config():
    json_file_path = os.path.join(os.path.dirname(__file__), 'data/servo_data.json')


class training_data_struct(dict):
    VALID_KEYS = [
        'face_location_box',
        'x_angle_adjustment',
        'y_angle_adjustment'
    ]
    
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
    
    def __setitem__(self, key, value):
        if key in self.VALID_KEYS:
            super().__setitem__(key, value)
        else:
            raise TypeError('Invalid key used: %s' % key)
    
    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]
        

class training_data_collection(list):
    def append(self, data: training_data_struct):
        super().append(data)
    

class servo_data_trainer():
    def __init__(self):
        self.data = training_data_collection()
    
    def load(self):
        with open(config.json_file_path, 'r', encoding='utf-8') as json_file:
            data_points = training_data_collection(json.load(json_file))
            for data_point in data_points:
                 self.data.append(
                    training_data_struct(data_point)
                 )
    
    def save_data(self):
        with open(config.json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, ensure_ascii=False, indent=4)


c = servo_data_trainer()
c.load()
c.data.append(
    training_data_struct(
        {
            'face_location_box': [1,2,3,4],
            'x_angle_adjustment': 1,
            'y_angle_adjustment': 1
        }
    )
)
# c.save_data()
# ^^^ data management is now set up.
# Now for the thing that will actually run the tests.

