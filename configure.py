import os
import ipdb
import configparser

"""
    parse default configuration and training configuration 
"""
class ConfigParser:
    def __init__(self):
        config_dir = os.path.dirname(__file__)
        self.default_config = configparser.RawConfigParser()
        self.file_path = os.path.join(config_dir, "config", "default.config")
        self.default_config.read(self.file_path)

    def get(self, field, name):
        return self.default_config.get(field, name)

    def getint(self, field, name):
        return self.default_config.getint(field, name)

    def getfloat(self, field, name):
        return self.default_config.getfloat(field, name)

    def getboolean(self, field, name):
        return self.default_config.getboolean(field, name)

    def set(self, field, name, value):
        self.default_config.set(field, name, value)
        file = open(self.file_path, "w", encoding="UTF-8")
        self.default_config.write(file)
        file.close()