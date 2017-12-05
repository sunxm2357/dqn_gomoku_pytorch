import os


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
