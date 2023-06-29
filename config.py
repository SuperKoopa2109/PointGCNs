# config.py
import configparser

class Configuration:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

    def get_value(self, section, option):
        return self.config.get(section, option)

    def set_value(self, section, option, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)

    def save(self):
        with open(self.file_path, 'w') as config_file:
            self.config.write(config_file)

# Create a global instance of the Configuration class
config = Configuration('config.ini')