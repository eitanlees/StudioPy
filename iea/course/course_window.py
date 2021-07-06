"""
Ezra S. Brooker
Date Created: 2021

Department of Scientific Computing
Florida State University

Course Window class that defines the GUI interface for a generic course.
Takes in a config file that gives window configuration details and the
module config filename and paths to know what modules to load for the 
selected course. This will enable multiple courses to be built and use
the same driving framework.

"""
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import importlib
import PySimpleGUI as sg

from iea.utils.base_window import BaseWindow
from iea.course.course_module import CourseModule


class CourseWindow(BaseWindow):

    title = "Generic Course Title"
    subwindows = {}

    def __init__(self,configfile,launch=False):
        super().__init__(configfile, launch)


    def _configure(self):
        """ If not using BaseWindow _configure, need to define custom one here """
        self._configure_window()
        self._configure_modules()
        self._configure_layout()


    def _configure_modules(self):
        """ Configure course modules """
        self.module_configs = self.config['module_configs']
        self.modules = {}
        self.submodules = {}
        self.location = self.config['location']
        for file in self.module_configs:
            newmodule = CourseModule(configfile=f"{self.location}/{file}")
            self.modules[newmodule.module] = newmodule
            if newmodule.submodules is not None:
                for key, submod in newmodule.submodules.items():
                    self.submodules[key]  = newmodule.location.replace('/','.')
                    self.submodules[key] += submod.replace('.py','')
        self.modkeys = list(self.modules.keys())


    def _configure_layout(self):
        """ Course window layout custom configuration """
        self.title = self.config['title']
        
        # Set up the course window BACK and EXIT buttons
        col_home = [[
            sg.Button("Exit", size=(30,1), key="-EXIT-") 
        ]]

        col_home += [ [mod.module_button] for mod in self.modules.values() ]
        col_mods  = [  mod.module_column  for mod in self.modules.values() ]

        self.layout     = [[ sg.Pane([sg.Column(col_home, key=f"{self.title}")]) ]]
        self.layout[0] += col_mods


    def launch(self):
        super().launch()
        if self.wincfg['finalize']:
            home_mod = list(self.modules.values())
            homepage = home_mod[0].colkey
            self.window[homepage].update(visible=True)
        self.__current_page = self.modkeys[0]


    def check_event(self,event,values):

        if event in self._EXIT_LIST + ['-EXIT-']:
            [w.window.close() for w in self.subwindows.values()]
            self.window.close() 
            return True

        elif event in self.modkeys:

            if event != self.__current_page:
                self.__reveal_column(self.modules[event].colkey)
                self.__current_page = event

        elif event in self.submodules.keys():
                imp_module = importlib.import_module(self.submodules[event])
            # try:
                submodule  = imp_module.SubModuleWindow()
                submodule.launch()
                self.subwindows[submodule.window] = submodule
            # except:
            #     sg.Popup(f"[Course Error] Submodule \"{self.submodules[event]}\" is missing!")

        return False


    def print_configuration(self):
        print(f"\nWindow Configuration:\n{self.wincfg}")
        print(f"\nCourse name: {self.title}\n")
        for module in self.modules.values():
            module.print_configuration()


    def __reveal_column(self,column):
        [self.window[mod.colkey].update(visible=False) for mod in self.modules.values()]
        self.window[column].update(visible=True)


if __name__ == '__main__':

    window = CourseWindow(configfile='config/acs2.yml')
    window.print_configuration()
    window.launch()
    window.event_loop()
