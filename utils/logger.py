import os
import pdb
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_args(logger, args):
    opt = vars(args)
    logger.cprint('------------ Options -------------')
    for k, v in sorted(opt.items()):
        logger.cprint('%s: %s' % (str(k), str(v)))
    logger.cprint('-------------- End ----------------\n')


def init_logger(log_dir, args, print=None):
    mkdir(log_dir)
    log_file = os.path.join(log_dir, 'log_%s.txt' %args.modes       )
    logger = IOStream(log_file)
    # logger.cprint(str(args))
    ## print arguments in format
    if print != None:
        print_args(logger, args)
    return logger