import logging
import os
from logging.handlers import RotatingFileHandler


level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_logger(
    name: str = None,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
    maxBytes: int = 1e8,
    backupCount: int = 1,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.
        maxBytes (int): The maximum size of the log file.
            Only used if log_path is not None.
        backupCount (int): Number of rolling backup log files.
            If log_path is `app.log` and backupCount is 3, we will have
            `app.log`, `app.log.1`, `app.log.2` and `app.log.3`.
            Only used if log_path is not None.
        ranked (bool): Get ranked_logger for lighting trianing if True.


    Note that console_level should only be used when configuring the
    root logger.
    """

    logger = logging.getLogger(name)
    if hasattr(logger, "root"):
        root_logger = logger.root
    else:
        root_logger = logger.logger.root
    root_logger.setLevel(logging.WARNING)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
    )

    # set up the logfile handler
    if log_path:
        if not os.path.isabs(log_path):
            log_path = os.path.join(os.environ["PROJECT_ROOT"], log_path)
        # logTime = datetime.datetime.now()
        # fn1, fn2 = os.path.splitext(log_path)
        # log_filename = f"{fn1}-{logTime.strftime('%Y%m%d-%H%M')}{fn2}"
        log_filename = log_path
        if os.path.dirname(log_filename):
            os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        # fh = logging.FileHandler(log_filename)
        fh = [
            handler
            for handler in root_logger.handlers
            if type(handler) == logging.FileHandler
        ]
        if fh:
            # global_logger.info(
            #     "Replaced the original root filehandler with new one."
            # )
            fh = fh[0]
            root_logger.removeHandler(fh)
        fh = RotatingFileHandler(
            filename=log_filename, maxBytes=maxBytes, backupCount=backupCount
        )
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    # set up the console/stream handler
    # if name and console_level:
    #     raise ValueError(
    #         "`console_level` should only be set when configuring root logger."
    #     )
    if console_level:
        sh = [
            handler
            for handler in root_logger.handlers
            if type(handler) == logging.StreamHandler
        ]
        if sh:
            # global_logger.info(
            #     "Replaced the original root streamhandler with new one."
            # )
            sh = sh[0]
            root_logger.removeHandler(sh)
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)
    return logger
