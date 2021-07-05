'''
Ezra S. Brooker
Date Created: 2021-07-05

Demonstration of Heat PDE.

'''

import sys
if sys.version[0] != '3':
    raise Exception("Python 2 is no longer supported, please use Python 3")

import os
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pdes.submodule_window import SubModuleWindowBase

class SubModuleWindow(SubModuleWindowBase):


  def __init__(self,configfile=None, launch_window=False):

    self.configfile = configfile
    self.module_layout()
    if launch_window:
      self.launch_window()

  def module_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme("Dark")

    buttons = [
      [ sg.Test("This WINDOW is currently under development", key="-DEVMSG-"), sg.Button("Exit", key="-EXIT-")                                                                               ],
      [ sg.Button("Submit", key="-SUBMIT-",bind_return_key=True,visible=False)                                                                                         ]
      ]

    canvas = [[sg.Canvas(key="-CANVAS-")]]

    self.layout = [
        [
          sg.Col(buttons),
          sg.VSeperator(),
          sg.Col(canvas)
        ]
    ]

  # def launch_window(self, begin_read=False):

  #     self.wincfg['finalize']  = True
  #     self.wincfg['return_keyboard_events'] = True
  #     self.wincfg['resizable'] = True
  #     self.wincfg["location"]  = [100,100]
  #     self._data()
  #     self._pad()
  #     self.window = sg.Window(self.title, self.layout, **self.wincfg)
  #     self._draw()
  #     if begin_read:
  #       self.read_window()

  # def read_window(self):
  #     while True:
  #         event, values = self.window.read()
  #         closed = self.check_read(event,values)
  #         if closed:
  #           break

  # def check_read(self,event,values):

  #   if event in self.__EXIT_LIST + ["-EXIT-"]:
  #     self.window.close()
  #     return True

  #   elif event == "-SUBMIT-":
  #     self._data()
  #     self._pad()


  #   self._draw()

  #   return False


    def _heat1d(self):
        pass

    def _heat2d(self):
        pass

    def _heat3d(self):
        pass




    
if __name__ == '__main__':

  sub = SubModuleWindow()
  sub.launch_window()
  sub.read_window()
