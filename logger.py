import logging


class Logger:

    def __init__(
            self,
            opt,
            logging_level=logging.INFO,
            file_level=logging.INFO,
            stream_level=logging.INFO
    ):
        self.opt = opt
        self.log_path = opt.log_path
        self.logging_level = logging_level

        self.file_level = file_level
        self.stream_level = stream_level

        self.logger = logging.getLogger('logger.log')
        self.logger.setLevel(self.logging_level)

        self.configure()

    def configure(self):
        log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(log_format)

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(log_format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
