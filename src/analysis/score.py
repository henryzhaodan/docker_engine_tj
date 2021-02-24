import numpy

from kernel.utils.util import AttentionHeadPoseSign, AttentionRealtimeSign, get_head_pose_score


def get_gaze_score(gaze):
    gaze_vector = numpy.deg2rad(gaze)
    lr_value = gaze_vector[0]  # 眼睛左右看指标
    tb_value = gaze_vector[1] - 0.05  # 眼睛上下看指标
    return lr_value, tb_value


class PostScore(object):
    def __init__(self, opts_post):
        self.opt_post = opts_post
        self.attention_head_pose_sign = AttentionHeadPoseSign(num_thred=self.opt_post.ATTENTION_HEAD_POSE_FRAME_THRED)
        self.attention_sign = AttentionRealtimeSign(num_thred=self.opt_post.ATTENTION_FRAME_THRED)
        self.head_pose_score_list = []  # 滑动窗口list
        self.attention_sign_list = []  # 注意力list

    def head_pose_avg_sign(self, pitch, roll):
        # head pose realtime score compute  坐姿实时值     参数为角度
        head_pose_realtime_score = get_head_pose_score(pitch, roll, self.opt_post.PITCH_NORMAL_ANGLE)
        # head_pose_score_max_len smooth  滑动窗口  当前帧之前N帧坐姿平均
        self.head_pose_score_list.append(head_pose_realtime_score)
        # print(head_pose_score_list)
        if len(self.head_pose_score_list) > self.opt_post.HEAD_POSE_SCORE_MAX_LEN:
            self.head_pose_score_list = self.head_pose_score_list[-self.opt_post.HEAD_POSE_SCORE_MAX_LEN:]
        head_pose_avg_score = numpy.mean(self.head_pose_score_list, axis=0)
        # False 持续N帧坐姿不正确     True默认坐姿正确
        attention_head_pose_sign = self.attention_head_pose_sign(
            head_pose_realtime_score, self.opt_post.ATTENTION_HEAD_POSE_SCORE_THRED)
        return head_pose_avg_score, attention_head_pose_sign

    def eye_gaze_avg(self, lr_value, tb_value, attention_head_pose_sign):

        # 已经预处理 持续N帧注意力不集中 输出False
        attention_realtime_sign = self.attention_sign(lr_value, tb_value,
                                                      self.opt_post.ATTENTION_ANGLE_THRED[0],
                                                      self.opt_post.ATTENTION_ANGLE_THRED[1])
        # head pose attention close eye   attention_sign_list modify
        if not attention_head_pose_sign or not attention_realtime_sign:
            self.attention_sign_list.append(0)
            if not attention_head_pose_sign:  # 在这之前 ATTENTION_HEAD_POSE_FRAME_THRED-1帧值 归0
                for i in range(2, self.opt_post.ATTENTION_HEAD_POSE_FRAME_THRED):
                    self.attention_sign_list[len(self.attention_sign_list) - i] = 0
            elif not attention_realtime_sign:  # 在这之前 ATTENTION_FRAME_THRED-1帧值 归0
                for i in range(2, self.opt_post.ATTENTION_FRAME_THRED):
                    self.attention_sign_list[len(self.attention_sign_list) - i] = 0
            else:  # 在这之前 3-1帧值 归0
                for i in range(2, 4):
                    self.attention_sign_list[len(self.attention_sign_list) - i] = 0
        else:
            self.attention_sign_list.append(1)

        # 滑动窗口  当前帧之前N帧注意力平均值
        if len(self.attention_sign_list) > self.opt_post.ATTENTION_SCORE_MAX_LEN:
            self.attention_sign_list = self.attention_sign_list[-self.opt_post.ATTENTION_SCORE_MAX_LEN:]
        attention_pose_avg_score = (numpy.sum(self.attention_sign_list, axis=0) / len(self.attention_sign_list)) * 100

        return attention_pose_avg_score

    def run(self, index, results):
        if results.head_pose is not None and len(results.head_pose) > 0:
            yaw, pitch, roll = results.head_pose[index]
            head_pose_avg_score, attention_head_pose_sign = self.head_pose_avg_sign(pitch, roll)
            lr_value, tb_value = get_gaze_score(results.gaze[index])
            attention_pose_avg_score = self.eye_gaze_avg(lr_value, tb_value, attention_head_pose_sign)
        else:
            head_pose_avg_score = 0.0
            attention_pose_avg_score = 0.0
        results.head_pose_score = head_pose_avg_score
        results.attention_score = attention_pose_avg_score
