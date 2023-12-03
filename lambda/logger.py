import logging
 
class Logger:
    def __init__(self, path = '', clevel = logging.INFO, Flevel = logging.INFO):
        if path == '':
            self.logger = logging.getLogger()
        else:
            self.logger = logging.getLogger(path)
        self.logger.setLevel(clevel)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        self.logger.addHandler(sh)

        if path != '':
            #设置文件日志
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(Flevel)
            self.logger.addHandler(fh)
 
    def debug(self,message):
        self.logger.debug(message)
 
    def info(self,message):
        self.logger.info(message)
 
    def warn(self,message):
        self.logger.warning(message)
    
    def warning(self, message):
        self.warn(message=message)
 
    def error(self,message):
        self.logger.error(message)
 
    def cri(self,message):
        self.logger.critical(message)
 
if __name__ =='__main__':
    logyyx = Logger('', logging.DEBUG, logging.INFO)
    logyyx.debug('一个debug信息')
    logyyx.info('一个info信息')
    logyyx.warn('一个warning信息')
    logyyx.error('一个error信息')
    logyyx.cri('一个致命critical信息')