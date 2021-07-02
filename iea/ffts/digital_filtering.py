'''
Author: Ezra S. Brooker
Date Created: 2021 July 02
Date Modified: 

Applied Computational Science II
Interactive Examples Applet
Dept of Scientific Computing
Florida State University

Proof-of-Concept for basic FFT examples using
the PySimpleGUI package for generating the GUI

Digital Filtering

'''

# Should already have
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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


from IPython import embed

class SubModuleWindow:

  title = "FFTs: Digital Filtering"

  __EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]
  __figure_agg = None

  __f      = "x*cos(x**2)"
  __nsamps = 16
  __ntrunc = 0
  __x      = np.linspace(0,1,__nsamps)
  __yi     = __x*np.cos(__x**2)
  __ys     = __x*np.cos(__x**2)
  __yp     = __x*np.cos(__x**2)
  __ft     = np.fft.rfft(__yp)

  wincfg = {}
  __xvar = symvar('x')

  def __init__(self,configfile=None, launch_window=False):

    self.configfile = configfile
    self.module_layout()
    if launch_window:
      self.launch_window()

  def module_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme('Dark')

    buttons = [
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Button("Reset", key='-RESET-'), sg.Button("Exit", key='-EXIT-')                                                   ],
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Text("Type a function f(x), press <ENTER>", size=(30,1))                                                          ],
      [ sg.InputText("x * cos(x**2)", size=(50,1), key="-IFUNC-")                                                            ],
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Button("Generate Random Noise", key="-GENRAN-"), sg.Button("Generate Systematic Noise", key="-GENSYS-")           ],
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Text("Number of Random Data Samples  ", size=(30,1))                                                              ], 
      [ sg.Combo([16,32,64,128], default_value=self.__nsamps, enable_events=True, key="-NSLIDE-")                            ],
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Text("Number of Truncated Frequencies", size=(30,1))                                                              ], 
      [ sg.Slider(range=(0,128,2), default_value=self.__ntrunc, orientation='h', enable_events=True, key="-TSLIDE-")         ],
      [ sg.Text("", size=(1,1)), sg.Button("Submit", visible=False, bind_return_key=True, key='-SUBMIT-'),                   ],
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

      self.wincfg['finalize'] = True
      self.window = sg.Window(self.title, self.layout, **self.wincfg)
      if begin_read:
        self.read_window()

  def read_window(self):

      while True:
          event, values = self.window.read()
          close_now = self.check_read(event,values)
          if close_now:
            break


  def check_read(self,event,values):

    if event in self.__EXIT_LIST + ['-EXIT-']:
      self.window.close()
      return True
 
    elif event == '-SUBMIT-':
      inpf = values['-IFUNC-']
      try:
        self.__f = sympify(inpf)
        self._generate_signal()
      except:
        self.__f = None
        sg.Popup("Invalid Function!")

    elif event == '-GENRAND-':
      self._generate_noise()        


    elif event == "-NSLIDE-":
      self.__nsamps = int(values['-NSLIDE-'])
      print("nsamps", self.__nsamps)
      self.__x      = np.linspace(0,1,self.__nsamps)
      self._generate_signal()
      self._generate_noise()
      self._smooth_signal()

    elif event == "-TSLIDE-":
      self.__ntrunc = int(values['-TSLIDE-'])
      if self.__ntrunc > self.__nsamps:
        self.__ntrunc = self.__nsamps
      self._smooth_signal()
      print("ntrunc", self.__ntrunc, (self.__nsamps-self.__ntrunc)//2)


    if type(self.__f) is not None:
      if self.__figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
          self.__figure_agg.get_tk_widget().forget()
          plt.close('all')
      self._smooth_signal()
      figr = self._make_plot()
      self.__figure_agg = self._draw_figure(self.window["-CANVAS-"].TKCanvas, figr)
  
    return False

  def _draw_figure(self, canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


  def _generate_signal(self):
    f = lambdify(self.__xvar,self.__f,"numpy")
    self.__yi = f(self.__x)
    self.__ys = f(self.__x)

  def _generate_noise(self):
    ym        = self.__yi.mean()
    yr        = np.random.random(self.__nsamps)
    self.__yp = ((np.random.random()*self.__yi+ym)-(np.random.random()*self.__yi-ym)) * yr + self.__yi-ym

  def _smooth_signal(self):
    self.__ft = np.fft.rfft(self.__yp)
    rft = np.copy(self.__ft)
    rft[(self.__nsamps-self.__ntrunc)//2:] = 0
    self.__ys = np.fft.irfft(rft)

  def _make_plot(self):

    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.plot(self.__x,self.__yi,label='Full', color='b')
    ax1.scatter(self.__x,self.__yp,label='Padded', s=10, color='k')

    # Plot both power spectra
    ax1.plot(self.__x,self.__yi, color='b')
    ax2.scatter(self.__x,self.__ys, s=10, color='k')

    # Add any plot frills you want
    ax1.set_xlabel('domain')
    ax1.set_ylabel("data")
    ax2.set_xlabel('domain')
    ax2.set_ylabel("data")
    ax1.legend(bbox_to_anchor=(1.05,1), loc='upper left', frameon=False)
    fig.tight_layout()
    return fig


if __name__ == '__main__':

  submodule = SubModuleWindow()
  submodule.launch_window()
  submodule.read_window()

