from analysis import InterResultSet


def turn_coco_box(boxes):
    bbox = []
    for box in boxes:
        if len(box) == 0:
            continue
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = box[0] + w / 2
        y = box[1] + h / 2
        bbox.append([x, y, w, h])
    return bbox


def sort_to_result_set(results, opts):
    sorted_results = InterResultSet()
    sorted_results.confidences = results['confidence'][opts.use_stacks[0][0]]
    sorted_results.boxes = results['box'][opts.use_stacks[0][0]]
    sorted_results.head_sign = len(sorted_results.boxes) > 0
    sorted_results.head_pose = results['headpose'][opts.use_stacks[1][0]]
    sorted_results.landmarks = results['landmark'][opts.use_stacks[2][0]]
    sorted_results.gaze = results['gaze'][opts.use_stacks[3][0]]
    sorted_results.inference_time = results['time']
    return sorted_results