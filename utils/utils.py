import logging
import time


def logout():
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    logging.basicConfig(filename='../logs/' + log_name,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger = logging.getLogger(log_name)
    logger.addHandler(console)

    return logger
