import os
CURRENT = os.path.split(os.path.abspath(__file__))[0]


class OptsPost(object):
    # -------test
    DEBUG = 2  # 0--没有调试显示, 1--显示人脸框和特征点等, 2--显示视线方向信息和头部姿态
    ATTENTION_FRAME_THRED = 5  # 眼球 连续5帧以上判断 不认真 False
    ATTENTION_ANGLE_THRED = [20, 15]  # 眼球左右 上下角度阈值  37, 20
    ATTENTION_HEAD_POSE_FRAME_THRED = 5  # 注意力判断需要头部姿态不正确连续的帧数阈值  连续5帧以上判断注意力不集中
    PITCH_NORMAL_ANGLE = -19  # 正常坐姿本身合理的角度   added by henry
    ATTENTION_HEAD_POSE_SCORE_THRED = 85  # 注意力判断需要头部姿态分值低于xx
    HEAD_POSE_SCORE_MAX_LEN = 100  # 头部姿态 100帧做一个统计
    ATTENTION_SCORE_MAX_LEN = 100  # 注意力 100帧做一个统计
    FRAMECOUNT = 150  # 150
    NOTSQL = 5
    MINIBOXSIDELENGTH = 160  # 距离远的框舍弃


opts_post = OptsPost()
