import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import os, pathlib, yaml
import importlib
import PySimpleGUI as sg

from course.course_layout import CourseLayout

class CourseWindow:

    __EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]

    submods = {}

    def __init__(self,configfile,launch_window=False):
        self.configfile = configfile
        self.configure()
        if launch_window:
            self.launch_window()

    def configure(self, configfile=None):

        if configfile is not None:
          self.configfile = configfile

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
            raise Exception(f"[ConfigFileRead Error] Can't read this file extension: {self.configfile}")

        self.wincfg = self.config['window config']
        self.layout = CourseLayout(self.configfile)

    def launch_window(self, begin_read=False):

        self.window = sg.Window(self.layout.course, self.layout.layout, **self.wincfg)

        if self.wincfg['finalize']:
            home_mod = list(self.layout.modules.values())
            homepage = home_mod[0].colkey
            self.window[homepage].update(visible=True)

        self.current_page = self.layout.modkeys[0]

        if begin_read:
            self.read_window()

    def read_window(self):

        while True:
            event, values = self.window.read()
            self.check_read(event,values)
        [w.window.close() for w in self.submods.values()]
        self.window.close()

    def reveal_column(self,column):
        [self.window[mod.colkey].update(visible=False) for mod in self.layout.modules.values()]
        self.window[column].update(visible=True)

    def print_configuration(self):

        print(f"\nWindow Configuration:\n{self.wincfg}")
        self.layout.print_configuration()

    def check_read(self,event,values):

        if event in self.__EXIT_LIST + ['-EXIT-']:
            return True

        elif event in self.layout.modkeys:

            if event != self.current_page:
                self.reveal_column(self.layout.modules[event].colkey)
                self.current_page = event

        elif event in self.layout.submodules.keys():
            imp_module = importlib.import_module(self.layout.submodules[event])
            try:
                submodule  = imp_module.SubModuleWindow(launch_window=True)
                self.submods[submodule.window] = submodule
            except:
                sg.Popup(f"[Course Error] Submodule \"{self.layout.submodules[event]}\" is missing!")

        return False

if __name__ == '__main__':

    window = CourseWindow(configfile='config/course.yml')
    window.print_configuration()
    window.launch_window(begin_read=True)
