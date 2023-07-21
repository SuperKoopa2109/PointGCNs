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


    def is_running_in_jupyter(self):
        try:
            # Check if the 'get_ipython' function exists
            shell = get_ipython().__class__.__name__

            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter Notebook or JupyterLab
            else:
                return False  # Other interactive shell
        except NameError:
            return False  # Not in an interactive shell

    def is_running_in_colab(self):
        try:
            # Check if the 'get_ipython' function exists

            RunningInCOLAB = 'google.colab' in str(get_ipython())

            if RunningInCOLAB:
                return True
            else:
                return False  # Other interactive shell
        except NameError:
            return False  # Not in an interactive shell

# Create a global instance of the Configuration class
param_config = Configuration('config.ini')