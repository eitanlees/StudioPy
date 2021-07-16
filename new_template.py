"""
Ezra S. Brooker

    Department of Scientific Computing
    Florida State University


"""

import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import numpy as np
import numpy.linalg as la

from iea.utils.base_window import BaseWindow

plt.style.use("dark_background")
pi = np.pi

class SubModuleWindow(BaseWindow):

  title="SubModuleWindow Template"

  _x = np.linspace(-np.pi,np.pi,500)
  _y = np.sin(_x)

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    buttons = [
      [ sg.Text("This is sg.Text ELEMENT", size=(20,1), key='-TEXT1-') ],
      [ sg.Button("This is sg.Button ELEMENT", size=(30,4), key='-BUTTON1-') ],
      [ sg.Text("", size=(1,1))], # Blank line creator
      [ sg.Combo([0,2,4,6,8,10], size=(10,6), key='-COMBO1-'), sg.Text("<-- sg.Combo") ],
      
      ]

    canvas = [[sg.Image(key="-IMAGE-")]]

    multi = [
      [sg.Multiline("Hi, this is an \nsg.Multiline ELEMENT \nused for multiple \nlines of text!", size=(15,10), key="-MULTI1-")],
      [ sg.Button("Exit", size=(10,3), key="-EXIT-") ],
    ]

    self.layout = [
        [
          sg.Col(buttons),
          sg.VSeperator(),
          sg.Col(canvas),
          sg.VSeperator(),
          sg.Col(multi)
        ]
    ]

  def launch(self):
      super().launch()
      self._draw()


  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 

  def _draw(self):

    # Add any plot frills you want

    plt.plot(self._x,self._y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Example Plot")
    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)




if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

