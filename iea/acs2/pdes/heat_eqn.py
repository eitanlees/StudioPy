"""
Ezra S. Brooker
Date Created: 2021-07-05

Department of Scientific Computing
Florida State University

Heat Equation Demonstration


"""

import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

from iea.utils.base_window import BaseWindow



class SubModuleWindow(BaseWindow):



  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch)


  def _configure_layout(self):
    super()._configure_layout()


  def launch(self):
      super().launch()


  def event_loop(self):
      while True:
          event, values = self.window.read()
          close_now = self.check_read(event,values)
          if close_now:
            break


if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

