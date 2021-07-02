'''
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

'''

# Should already have
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import OrderedDict
import tempfile
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

class SubModuleClose(Exception):
  print("[FFTs: Padding Sequences] Window closing")  

class SubModuleWindow:

  title = "FFTs: Padding Sequences"

  __EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]
  __figure_agg = None

  __choices = ['zeros','constant','linear2zero','linear2periodic']
  __textchs = ['Zeros', "Constant", "Linear to Zero", "Linear Periodic"]
  __textdct = dict(zip(__textchs,__choices))

  wincfg = {}
  __x = symvar('x')

  def __init__(self,configfile=None, launch_window=False):

    self.configfile = configfile
    self.module_layout()
    if launch_window:
      self.launch_window()

  def module_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme('Dark')

    buttons = [
      [ sg.Button("Submit", key='-SUBMIT-'), sg.Button("Generate Random Data", key="-RAND-"), sg.Button("Exit", key='-EXIT-')                ],
      [ sg.Text("Function f(x)",    size=(15,1)), sg.InputText("x * cos(x**2)", key="-INPUT-FUNC-")                                          ],
      [ sg.Text("Number of points", size=(15,1)), sg.InputText("33", key="-INPUT-NUM-")                                                      ],
      [ sg.Text("Padding Method",   size=(15,1)), sg.Combo(self.__textchs, size=(20,20), key="-INPUT-PAD-", default_value=self.__textchs[0]) ],
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
          self.check_read(event,values)
      self.window.close()


  def check_read(self,event,values):

    if event in self.__EXIT_LIST + ['-EXIT-']:
      self.window.close()
      return True

    elif event == '-SUBMIT-':
      inpf = values['-INPUT-FUNC-']
      inpt = self.__textdct[values['-INPUT-PAD-']]
      inpn = int(values['-INPUT-NUM-'])
      expr = sympify(inpf)
      func = lambdify(self.__x,expr,"numpy")
      figr = self._plot_fft(func,n=inpn,padding=inpt)
      if self.__figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
          self.__figure_agg.get_tk_widget().forget()
          plt.close('all')

      self.__figure_agg = self._draw_figure(self.window["-CANVAS-"].TKCanvas, figr)

    elif event == '-RAND-':
      pass
  
    return False

  def _draw_figure(self, canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


  def _plot_fft(self, f,n=33,padding='zeros'):

    # Spatial domain for data
    xf = np.linspace(0,1,64)

    # Create full dataset
    yf = f(xf)

    # Truncated dataset with "n" points
    yp = yf[:n]

    # Difference in full and truncated series
    nd = xf.size - n

    # Now pad the truncated the sequence
    if padding.lower() == 'zeros':
      ya = np.zeros(nd)

    elif padding.lower() == 'constant':
      ya = np.ones(nd)*yp[-1]
      
    elif padding.lower() == 'linear2zero':
      ya = np.interp(xf[n:],[xf[n-1],xf[-1]], [yp[-1],0.0])
      
    elif padding.lower() == 'linear2periodic':
      ya = np.interp(xf[n:],[xf[n-1],xf[-1]], [yp[-1],yp[0]])
      
    yp = np.append(yp,ya)
    del(ya)

    # Compute FFT and power spectra of both sequences
    fps = np.abs(np.fft.fft(yf))**2 # original
    pps = np.abs(np.fft.fft(yp))**2 # padded

    fps[fps < 2**(-6)] = 2**-6
    pps[pps < 2**(-6)] = 2**-6

    # Compute FFT frequences and sort them
    freq = np.fft.fftfreq(yf.size,1.0/yf.size)
    idx = np.argsort(freq)


    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.scatter(xf,yf,label='Full', s=10, color='b')
    ax1.scatter(xf,yp,label='Padded', s=10, color='k')

    # Plot both power spectra
    ax2.scatter(freq[idx],fps[idx], s=10, color='b')
    ax2.scatter(freq[idx],pps[idx], s=10, color='k')

    # Add any plot frills you want
    ax1.set_xlabel('spatial domain')
    ax1.set_xlim(left=0.0)
    ax1.set_ylabel("data")
    ax2.set_xlabel('frequency')
    ax2.set_xlim(left=0.0)
    ax2.xaxis.set_ticks(np.arange(0,yf.size//2+1,8))
    ax2.set_ylabel("power spectra")
    ax2.set_yscale('log',basey=2)
    ax2.yaxis.set_ticks([2**-6, 2**0, 2**6, 2**12])
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels = ['0', r'$2^0$', r'$2^6$', r'$2^{12}$']
    ax2.set_yticklabels(labels)

    ax1.legend(bbox_to_anchor=(1.05,1), loc='upper left', frameon=False)
    fig.tight_layout()
    return fig



if __name__ == '__main__':

  window = SubModuleWindow()
  window.launch_window()
  window.read_window()
