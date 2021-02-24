class InterResultSet:
    def __init__(self):
        self.confidences = []
        self.boxes = []
        self.head_sign = False
        self.head_pose = []
        self.landmarks = []
        self.gaze = []
        self.inference_time = []
        self.index = -1
        self.head_pose_score = 0.00
        self.attention_score = 0.00


class Result:
    def __init__(self, addr):
        self.addr = addr
        self.data = []

    def to_json(self):
        return {
            'addr': self.addr,
            'data': self.data
        }


class Data:
    def __init__(self, frame, human, ats):
        self.frame = frame
        self.human = human
        self.head = None
        self.box = None
        self.ats = ats

    def to_json(self):
        return {
            'frame': self.frame,
            'human': self.human,
            'head': self.head,
            'box': self.box,
            'as': self.ats
        }


class Head:
    def __init__(self, y, p, r, s):
        self.y = y
        self.p = p
        self.r = r
        self.s = s

    def to_json(self):
        return {
            'y': self.y,
            'p': self.p,
            'r': self.r,
            's': self.s
        }


class Box:
    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.gaze = None

    def to_json(self):
        return {
            'x': self.x,
            'y': self.y,
            'h': self.h,
            'w': self.w,
            'gaze': self.gaze
        }


class Gaze:
    def __init__(self, y, p, e):
        self.y = y
        self.p = p
        self.e = e

    def to_json(self):
        return {
            'y': self.y,
            'p': self.p,
            'e': self.e
        }


def to_json(obj):
    return obj.to_json()
