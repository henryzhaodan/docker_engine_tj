from analysis import sort_to_result_set, turn_coco_box, Head, PostScore, Box, Gaze, Data
from configs import opts_post
from kernel import ModelInference
from kernel.utils import choose_box


def pack_results(results):
    if results.head_sign and len(results.head_pose) > 0:
        yaw, pitch, roll = results.head_pose[results.index]
        head = Head(round(yaw, 2), round(pitch, 2), round(roll, 2), round(results.head_pose_score, 2))
        bbox = turn_coco_box(results.boxes)
        if bbox is not None and len(bbox) > 0:
            box_x, box_y, box_h, box_w = bbox[0]
            box = Box(round(box_x, 2), round(box_y, 2), round(box_h.item(), 2), round(box_w.item(), 2))
        else:
            box = Box(0.00, 0.00, 0.00, 0.00)
        if results.gaze is not None:
            gaze_yaw, gaze_pitch = results.gaze[results.index]
        else:
            gaze_yaw = 0.00
            gaze_pitch = 0.00
        gaze = Gaze(round(gaze_yaw, 2), round(gaze_pitch, 2), 0)
        box.gaze = gaze
    else:
        results.attention_score = 0.00
        head = None
        box = None

    data = Data(-1, results.head_sign, round(results.attention_score, 2))
    data.head = head
    data.box = box
    return data


class AnalysisHelper:
    def __init__(self, opts):
        self.opt = opts
        self.opt_post = opts_post
        self.model_inference = ModelInference(self.opt)
        self.post_score = PostScore(opts_post)

    def process_image(self, image):
        height, width, _ = image.shape
        results = self.model_inference.run(image)
        results = sort_to_result_set(results, self.opt)
        index, _ = choose_box(image, width / 2, height / 2, results.boxes, self.opt_post.MINIBOXSIDELENGTH)
        self.post_score.run(index, results)

        if results.head_pose_score is None:
            results.head_pose_score = 0.0
            results.attention_score = 0.0

        return results

