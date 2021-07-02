import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import os, pathlib, yaml
import PySimpleGUI as sg

from course_module import CourseModule

class CourseLayout:

  def __init__(self, configfile):
    self.configfile = configfile
    self.configure()


  def configure(self, configfile=None):

    if configfile is not None:
      self.configfile = configfile

    if os.path.isfile(self.configfile):
      extension = pathlib.Path(self.configfile).suffix.strip('.')
    else:
      raise Exception(f"[CourseLayout Error] Module list file does not exist: {self.configfile}")

    with open(self.configfile, "r") as f:
      if extension in ("yaml", "yml"):
        try:
          self.config = yaml.safe_load(f.read())
        except yaml.YAMLError as exc:
          print(exc)
      else:
        raise Exception(f"[ConfigFileRead Error] Can't read this file extension: {self.configfile}")

    self.course = self.config['course']
    self.configure_modules()
    self.configure_layout()


  def configure_layout(self):

    # Set up the course window BACK and EXIT buttons
    col_home = [[
      sg.Button("Exit", size=(30,1), key="-EXIT-") 
    ]]

    col_home += [ [mod.module_button] for mod in self.modules.values() ]
    col_mods  = [  mod.module_column  for mod in self.modules.values() ]

    self.layout     = [[ sg.Pane([sg.Column(col_home, key=f"{self.course}")]) ]]
    self.layout[0] += col_mods


  def configure_modules(self):

    self.module_configs = self.config['module_configs']
    self.modules = {}
    self.submodules = {}
    
    for file in self.module_configs:
      
      newmodule = CourseModule(configfile=file)
      self.modules[newmodule.module] = newmodule
      if newmodule.submodules is not None:
        for key, submod in newmodule.submodules.items():
          self.submodules[key]  = newmodule.location.replace('/','.')
          self.submodules[key] += submod.replace('.py','')

    self.modkeys = list(self.modules.keys())

  def print_configuration(self):

    print(f"\nCourse name: {self.course}\n")

    for module in self.modules.values():
      module.print_configuration()


if __name__ == '__main__':

    course = CourseLayout(configfile='config/acs2.yml')
    course.print_configuration()
