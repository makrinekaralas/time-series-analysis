import logging
import sys

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

class Logger(object):
    """Logger class for logging

    Attributes
    ----------
    f_handler: FileHandler
        The file handler

    Methods
    ----------
    info()
       loggs INFO messages
    warning()
       loggs warnings
    error()
       loggs ERRORS
    exception()
       loggs exceptions
    """
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        # Create handlers
        f_handler = logging.FileHandler('tsa_log.log')
        # Configure handlers
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s'))
        # Add handlers to the logger
        self.logger.addHandler(f_handler)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)
