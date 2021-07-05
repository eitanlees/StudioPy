"""
Author: Ezra S. Brooker
Date Created: 2021 June 18
Date Modified: 

Applied Computational Science II
Interactive Examples Applet
Dept of Scientific Computing
Florida State University

Proof-of-Concept for basic FFT examples using
the PySimpleGUI package for generating the GUI

Padding Sequences

"""

# Should already have
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import os
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sympy import var as symvar
from sympy import sympify
from sympy.utilities.lambdify import lambdify

class SubModuleWindow:

  title = "FFTs: Padding Sequences"

  __EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]
  __figure_agg = None

  __choices = ["zeros","constant","linear2zero","linear2periodic"]
  __textchs = ["Zeros", "Constant", "Linear to Zero", "Linear Periodic"]
  __textdct = dict(zip(__textchs,__choices))

  __funcstr = "x*cos(x**2)"
  __padding = "Zeros"
  __nsamps  = 33

  wincfg = {}
  __x = symvar("x")

  def __init__(self,configfile=None, launch_window=False):

    self.configfile = configfile
    self.module_layout()
    if launch_window:
      self.launch_window()

  def module_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme("Dark")

    buttons = [
      [ sg.Button("Generate Random Data", key="-RAND-"), sg.Button("Exit", key="-EXIT-")                                                                               ],
      [ sg.Text("Function f(x): PRESS <ENTER>",   size=(25,1)), sg.InputText(self.__funcstr, key="-INPUT-FUNC-", size=(20,1))                                          ],
      [ sg.Text("Number of points", size=(25,1)), sg.Combo([i for i in range(1,65)], key="-INPUT-NUM-", enable_events=True, default_value=self.__nsamps, size=(20,20)) ],
      [ sg.Text("Padding Method",   size=(25,1)), sg.Combo(self.__textchs, size=(20,20), key="-INPUT-PAD-", default_value=self.__padding, enable_events=True)          ],
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

  def launch_window(self, begin_read=False):

      self.wincfg['finalize']  = True
      self.wincfg['return_keyboard_events'] = True
      self.wincfg['resizable'] = True
      self.wincfg["location"]  = [100,100]
      self._data()
      self._pad()
      self.window = sg.Window(self.title, self.layout, **self.wincfg)
      self._draw()
      if begin_read:
        self.read_window()

  def read_window(self):
      while True:
          event, values = self.window.read()
          closed = self.check_read(event,values)
          if closed:
            break

  def check_read(self,event,values):

    self.__funcstr = values["-INPUT-FUNC-"]
    self.__padding = self.__textdct[values["-INPUT-PAD-"]]
    self.__nsamps  = int(values["-INPUT-NUM-"]) - 1

    if event in self.__EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True

    elif event == "-SUBMIT-":
      self._data()
      self._pad()

    elif event in ("-INPUT-PAD-","-INPUT-NUM-"):
      self._pad()

    elif event == "-RAND-":
      self.__f = lambda x: np.random.random(x.size)
      self._data(user_func=False)
      self._pad()

    self._draw()

    return False

  def _draw(self):
    figure = self._plot_fft()
    if self.__figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
        self.__figure_agg.get_tk_widget().forget()
        plt.close("all")
    figure_canvas_agg = FigureCanvasTkAgg(figure, self.window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    self.__figure_agg =  figure_canvas_agg


  def _data(self, user_func=True):
    if user_func:
      try:
        expr      = sympify(self.__funcstr)
        self.__f  = lambdify(self.__x,expr,"numpy")
      except:
        sg.Popup("[Error] Invalid function given!")
    self.__xf = np.linspace(0,1,64)    # Spatial domain for data
    self.__yf = self.__f(self.__xf)    # Create full dataset


  def _pad(self):
    # Now pad the truncated the sequence
    self.__yp = self.__yf.copy()
    nd = self.__xf.size - self.__nsamps

    if self.__padding.lower() == "zeros":
      self.__yp[self.__nsamps:] = np.zeros(nd)

    elif self.__padding.lower() == "constant":
      self.__yp[self.__nsamps:] = np.ones(nd)*self.__yp[-1]
      
    elif self.__padding.lower() == "linear2zero":
      self.__yp[self.__nsamps:] = np.interp(self.__xf[self.__nsamps:], 
                                            [self.__xf[self.__nsamps], self.__xf[-1]], 
                                            [self.__yp[self.__nsamps], 0.0])
      
    elif self.__padding.lower() == "linear2periodic":
      self.__yp[self.__nsamps:] = np.interp(self.__xf[self.__nsamps:], 
                                            [self.__xf[self.__nsamps], self.__xf[-1]], 
                                            [self.__yp[self.__nsamps], self.__yp[0]])


  def _plot_fft(self):

    # Compute FFT and power spectra of both sequences
    fps = np.abs(np.fft.fft(self.__yf))**2 # original
    pps = np.abs(np.fft.fft(self.__yp))**2 # padded

    fps[fps < 2**(-6)] = 2**-6
    pps[pps < 2**(-6)] = 2**-6

    # Compute FFT frequences and sort them
    freq = np.fft.fftfreq(self.__yf.size,1.0/self.__yf.size)
    idx = np.argsort(freq)

    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.scatter(self.__xf,self.__yf,label="Full", s=10, color="b")
    ax1.scatter(self.__xf,self.__yp,label="Padded", s=10, color="k")

    # Plot both power spectra
    ax2.scatter(freq[idx],fps[idx], s=10, color="b")
    ax2.scatter(freq[idx],pps[idx], s=10, color="k")

    # Add any plot frills you want
    ax1.set_xlabel("spatial domain")
    ax1.set_xlim(left=0.0)
    ax1.set_ylabel("data")
    ax2.set_xlabel("frequency")
    ax2.set_xlim(left=0.0)
    ax2.xaxis.set_ticks(np.arange(0,self.__yf.size//2+1,8))
    ax2.set_ylabel("power spectra")
    ax2.set_yscale("log",base=2)
    ax2.yaxis.set_ticks([2**-6, 2**0, 2**6, 2**12])
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels = ["0", r"$2^0$", r"$2^6$", r"$2^{12}$"]
    ax2.set_yticklabels(labels)

    ax1.legend(bbox_to_anchor=(1.05,1), loc="upper left", frameon=False)
    fig.tight_layout()
    return fig

if __name__ == "__main__":

  window = SubModuleWindow()
  window.launch_window()
  window.read_window()
