import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import os, pathlib, yaml
import PySimpleGUI as sg

from IPython import embed

class CourseModule:

  def __init__(self, configfile):
    self.configfile = configfile
    self.configure()


  def configure(self, configfile=None):

    if configfile is not None:
      self.configfile        = configfile
      self.module_button     = None
      self.module_text       = None
      self.submodule_buttons = None

    if os.path.isfile(self.configfile):
      extension = pathlib.Path(self.configfile).suffix.strip('.')
    else:
      raise Exception(f"[CourseWindow Error] Module list file does not exist: {self.configfile}")

    with open(self.configfile, "r") as f:
      if extension in ("yaml", "yml"):
        try:
          self.config = yaml.safe_load(f.read())
        except yaml.YAMLError as exc:
          print(exc)
      else:
        sys.exit(f"SIGH: Can't read this file extension: {self.configfile}")

    if self.config['location'] is not None:
      self.location = self.config['location']
      
    self.setup_module_button()
    self.setup_module_text()
    self.setup_submodule_buttons()
    self.setup_module_column()


  def print_configuration(self):
    print(f"\nModule name: {self.module}")
    print(f"\n{self.text}\n")
    print(f"Submodules:  {self.submodules}\n")


  def setup_module_button(self):
    self.module = self.config['module']
    if self.module is None:
      self.module = 'Generic Button'
    self.module_button = sg.Button(self.module, size=(30,1), key=self.module)


  def setup_module_text(self):
    self.text = self.config['text']
    if self.text is None:
      self.text = "Some generic placeholder text"
    self.module_text = [[sg.Text(self.text)]]


  def setup_submodule_buttons(self):
    self.submodules = self.config['submodules']

    if self.submodules is not None:
      self.submodule_buttons = [ [sg.Button(submod, size=(30,1), key=submod)] for submod in self.submodules.keys()]
    else:
      self.submodule_buttons = []


  def setup_module_column(self):
    self.colkey = f"{self.module} Column"
    self.module_column = sg.Column(self.module_text + self.submodule_buttons, visible=False, key=self.colkey)


if __name__ == '__main__':

  cm = CourseModule(configfile='config/intro.yml')
  cm.print_configuration()

  cm = CourseModule(configfile='config/fft.yml')
  cm.print_configuration()

  mod = cm.submodule_check_read(list(cm.submodules.keys())[1], ())
  embed()
