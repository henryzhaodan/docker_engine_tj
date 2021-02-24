import sys
import logging
import cv2
import json
import nsq
import numpy
import os
import oss2
import torch
import requests
import uuid
from torch.multiprocessing import Process, Queue, Pool, set_start_method
import time
import toml

import handlers
from analysis import Result, to_json

from analysis.helper import AnalysisHelper, pack_results
from configs import opts

NSQ_READ_BUFFER = []
NSQ_WRITER = None
CONFIG_FILENAME = 'config.toml'
LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def load_config():
    return toml.load(CONFIG_FILENAME)


def config_logger(config, name):
    level = LOG_LEVEL_DICT.get(config['log']['level'].lower())
    logging.basicConfig(level=level, stream=sys.stdout,
                        format='[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s', datefmt="%Y-%m-%dT%H:%M:%S%z")
    logging.addLevelName(logging.DEBUG, 'DBG')
    logging.addLevelName(logging.INFO, 'INF')
    logging.addLevelName(logging.WARNING, 'WRN')
    logging.addLevelName(logging.ERROR, 'ERR')
    logging.addLevelName(logging.CRITICAL, 'CRT')
    return logging.getLogger(name)


def process_message(queue, config, message):
    logger = config_logger(config, 'reader')
    global NSQ_READ_BUFFER
    message.enable_async()
    # cache the message for later processing
    NSQ_READ_BUFFER.append(message)
    if len(NSQ_READ_BUFFER) >= 1:
        for msg in NSQ_READ_BUFFER:
            logger.debug('Get message in the NSQ reading queue: {}'.format(msg))
            queue.put(msg.body)
            msg.finish()
        NSQ_READ_BUFFER = []
    else:
        print('deferring processing')


def process_address(config, logger, bucket, address):
    return download_from_oss(config, logger, bucket, address)


def init_engine():
    opt = opts
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bin_set_p = opt.bin_set_p
    opt.bins_p = numpy.array(range(bin_set_p[0], bin_set_p[1], bin_set_p[2]))
    opt.num_bins_p = opt.bins_p.shape[0] + 1
    bin_set_g = opt.bin_set_g
    opt.bins_g = numpy.array(range(bin_set_g[0], bin_set_g[1], bin_set_g[2]))
    opt.num_bins_g = opt.bins_g.shape[0] + 1
    return opt


def process_video(config, opt, logger, file, addr):
    analysis_helper = AnalysisHelper(opt)
    frame_sample_interval = config['analysis']['sample_interval']
    video = cv2.VideoCapture(file)
    result = Result(addr)
    title_image = None
    title_image_file = None
    try:
        logger.debug('Analyzing video: {}...'.format(addr))
        num = 0
        while True:
            ret, image = video.read()
            if not ret:
                break
            if title_image is None:
                title_image = image
            num += 1
            if num % frame_sample_interval != 0:
                continue
            result_set = analysis_helper.process_image(image)
            data = pack_results(result_set)
            data.frame = num
            result.data.append(data)
        logger.debug('Exporting image...'.format(addr))
        title_image_file = '{}_title.png'.format(file)
        title_image = cv2.resize(title_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(title_image_file, title_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except Exception as e:
        logger.error('An error occurred in engine: {}'.format(e))
    finally:
        video.release()
    return result, title_image_file


def read_func(config, queue):
    logger = config_logger(config, 'reader')
    reading_handler = handlers.ReadingMessageHandler(process_message, config, queue)
    nsqd_tcp_address = config['nsq']['reader']['tcp_addr']
    nsqd_topic = config['nsq']['reader']['topic']
    nsqd_channel = config['nsq']['reader']['channel']
    nsq.Reader(message_handler=reading_handler,
               nsqd_tcp_addresses=nsqd_tcp_address,
               topic=nsqd_topic,
               channel=nsqd_channel,
               max_in_flight=config['nsq']['reader']['max_in_flight'])
    logger.info('Listening NSQ address={}, topic={}, channel={}...'
                .format(nsqd_tcp_address, nsqd_topic, nsqd_channel))
    nsq.run()


def write_func(config, queue, bucket):
    logger = config_logger(config, 'writer')
    http_url = config['nsq']['writer']['http_addr']
    while True:
        result, image = queue.get()
        addr = result.addr
        json_str = json.dumps(result, default=to_json)
        remote_result_file, remote_title_image_file = upload_to_oss(config, logger, bucket, addr, json_str, image)
        if remote_result_file is not None:
            requests.post(http_url, remote_result_file)


def compute_func(config, flow_id, opt, bucket, addr):
    logger = config_logger(config, '~{}'.format(flow_id[:6]))
    logger.info('Task begin')
    local_file = process_address(config, logger, bucket, addr)
    if local_file is None:
        return
    try:
        analysis_result, title_image_file = process_video(config, opt, logger, local_file, addr)
        logger.info('Task completed')
    except Exception as e:
        analysis_result = None
        title_image_file = None
        logger.error('An error occurred during analysis: {}'.format(e))
        logger.info('Task failed')
    finally:
        os.remove(local_file)
    return analysis_result, title_image_file


def get_random_id():
    return uuid.uuid4().hex


def dispatch_func(config, opt, reading_queue, writing_queue, bucket):
    logger = config_logger(config, 'dispatch')
    while True:
        raw_video = reading_queue.get().decode()
        flow_id = get_random_id()
        logger.info('Start new task for \'{}\''.format(raw_video))
        analysis_result, title_image_file = compute_func(config, flow_id, opt, bucket, raw_video)
        if analysis_result:
            writing_queue.put((analysis_result, title_image_file))
        else:
            logger.error('\'{}\' failed to parse and will be added to the queue again and wait for another attempt'
                         .format(raw_video))
            reading_queue.put(raw_video.encode())


def download_from_oss(config, logger, bucket, address):
    try:
        local_filename = logger.name
        local_file = os.path.join(config['oss']['download']['tmp_dir'], local_filename)
        bucket.get_object_to_file(address, local_file)
        logger.debug('Downloaded raw video from address: {}'.format(address))
        return local_file
    except Exception as e:
        logger.error('An error occurred during downloading: {}'.format(e))
        return None


def upload_to_oss(config, logger, bucket, addr, content, title_image):
    try:
        raw_filename = os.path.splitext(addr)[0]
        remote_result_file = add_file_suffixes(raw_filename, config['oss']['upload']['result_suffixes'])
        remote_title_image_file = add_file_suffixes(raw_filename, config['oss']['upload']['title_image_suffixes'])
        bucket.put_object(key=remote_result_file, data=content)
        logger.debug('Uploaded result data to address: {}'.format(remote_result_file))
        bucket.put_object_from_file(key=remote_title_image_file, filename=title_image)
        logger.debug('Uploaded title image to address: {}'.format(remote_title_image_file))
        return remote_result_file, remote_title_image_file
    except Exception as e:
        logger.error('An error occurred during uploading: {}'.format(e))
        return None, None
    finally:
        os.remove(title_image)


def add_file_suffixes(filename, suffixes):
    for suffix in suffixes:
        filename = '{}.{}'.format(filename, suffix)
    return filename


if __name__ == '__main__':
    global_config = load_config()
    main_logger = config_logger(global_config, 'main')
    main_logger.info('{} loaded.'.format(CONFIG_FILENAME))
    cv_opt = init_engine()
    set_start_method('spawn', force=True)
    main_logger.info('Initialized engine.')
    raw_video_queue = Queue()
    analysis_result_queue = Queue()
    main_logger.info('Initialized queues.')
    oss_auth = oss2.Auth(global_config['oss']['auth']['id'], global_config['oss']['auth']['secret'])
    oss_bucket = oss2.Bucket(oss_auth, global_config['oss']['endpoint'], global_config['oss']['bucket'])
    main_logger.info('Initialized oss connection.')
    reader_process = Process(target=read_func,
                             args=(global_config, raw_video_queue,),
                             daemon=True)
    writer_process = Process(target=write_func,
                             args=(global_config, analysis_result_queue, oss_bucket,),
                             daemon=True)
    reader_process.start()
    main_logger.info('Reading process started.')
    writer_process.start()
    main_logger.info('Writing process started.')
    analysis_process_pool = Pool(global_config['analysis']['max_parallels'], dispatch_func, (
        global_config, cv_opt, raw_video_queue, analysis_result_queue, oss_bucket,
    ))
    main_logger.info('Analysis process pool started.')
    while True:
        pass
    # dispatch_func(global_config, cv_opt, raw_video_queue, analysis_result_queue, oss_bucket)
