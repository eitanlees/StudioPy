"""
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

"""

# Should already have
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import os
import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
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

  title = "FFTs: Digital Filtering"

  __EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]
  __figure_agg = None

  __xvar  = symvar("x")
  __fstr  = "x*cos(x**2)"
  __nsamp = 16
  __ntrnc = 0
  __x     = np.linspace(0,1,__nsamp)
  __yi    = __x*np.cos(__x**2)
  __ys    = __x*np.cos(__x**2)
  __yp    = __x*np.cos(__x**2)
  __ft    = np.fft.rfft(__yp)
  __snr   = 0.01

  wincfg = {}

  def __init__(self,configfile=None, launch_window=False):

    self.configfile = configfile
    self.module_layout()
    if launch_window:
      self.launch_window()

  def module_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme("Dark")
    perc  = [0,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,98,99,100]
    snr   = [0.01,0.1,1,10,100]
    samp  = [16,32,64,128]
    b1,b2 = "Random Noise", "Systematic Noise"
    ran_d = { "key":"-RAND-"}
    sys_d = { "key":"-SYST-"}
    txt_d = { "size":(30,1) }
    exi_d = { "size":( 5,1), "key":"-EXIT-"  }
    fun_d = { "size":(31,1), "key":"-IFUNC-" }
    snr_d = { "default_value":self.__snr,   "enable_events":True, "key":"-SNR-",   "size":(4,5) }
    smp_d = { "default_value":self.__nsamp, "enable_events":True, "key":"-SAMP-",  "size":(4,5) }
    trn_d = { "default_value":self.__ntrnc, "enable_events":True, "key":"-TRUNC-", "size":(4,5) }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }
    buttons = [
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Text("<ENTER> a function --> f(x)", **txt_d)                         ],
      [ sg.InputText(self.__fstr, **fun_d)                                      ],
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Button(b1,  **ran_d), sg.Button(b2, **sys_d)                         ],
      [ sg.Combo(snr,  **snr_d), sg.Text("Signal-to-Noise Ratio", **txt_d)      ],
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Combo(samp, **smp_d), sg.Text("# of Random Samples", **txt_d)        ],
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Combo(perc, **trn_d), sg.Text("% of Frequencies Truncated", **txt_d) ],
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Button("Exit", **exi_d), sg.Button("Submit", **sub_d),               ],
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

      self.wincfg["finalize"]  = True
      self.wincfg["return_keyboard_events"] = True
      self.wincfg["resizable"] = True
      self.wincfg["location"]  = [100,100]
      self._generate_signal()
      self._smooth_signal()
      self.window = sg.Window(self.title, self.layout, **self.wincfg)
      self._draw()
      if begin_read:
        self.read_window()

  def read_window(self):
      while True:
          event, values = self.window.read()
          close_now = self.check_read(event,values)
          if close_now:
            break

  def check_read(self,event,values):

    self.__nsamp = int(values["-SAMP-"])
    self.__ntrnc = int(values["-TRUNC-"])

    if event in self.__EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event == "-SUBMIT-":
      if self.__fstr != values["-IFUNC-"]:
        self.__fstr = values["-IFUNC-"]
        self._generate_signal()
        self._random_noise()
        self._smooth_signal()
        self._draw()

    elif event == "-RAND-":
      self._random_noise()
      self._smooth_signal()
      self._draw()

    elif event == "-SYST-":
      self._systematic_noise()
      self._smooth_signal()
      self._draw()

    elif event == "-SAMP-":
      self.__x = np.linspace(0,1,self.__nsamp)
      self._generate_signal()
      self._random_noise()
      self._smooth_signal()
      self._draw()

    elif event == "-TRUNC-":
      self._smooth_signal()
      self._draw()

    elif event == "-SNR-":
      self._random_noise()
      self._smooth_signal()
      self._draw()

    return False


  def _draw(self):
    figure = self._make_plot()
    if self.__figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
        self.__figure_agg.get_tk_widget().forget()
        plt.close("all")
    figure_canvas_agg = FigureCanvasTkAgg(figure, self.window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    self.__figure_agg =  figure_canvas_agg

  def _generate_signal(self):
    try:
      self.__f = sympify(self.__fstr)
      f = lambdify(self.__xvar,self.__f,"numpy")
    except:
      sg.Popup("[Error] Invalid Function!")
    self.__yi = f(self.__x)
    self.__ys = f(self.__x)
    self.__yp = f(self.__x)

  def _random_noise(self):
    signal     = np.mean(self.__yi**2)
    sigavg     = 10*np.log10(signal)
    noises     = np.sqrt(10**((sigavg - self.__snr)*0.1))
    self.__yp  = self.__yi + np.random.normal(0.e0, noises, self.__nsamp)
    self.__ft  = np.fft.rfft(self.__yp)

  def _systematic_noise(self):
    self._random_noise()
    signal = np.mean(self.__ft**2)
    sigavg = 10*np.log10(signal)
    noises = np.sqrt(10**((sigavg - self.__snr*0.1)*0.1))
    highs  = int(0.85*self.__ft.size)
    self.__ft[highs:] += np.random.normal(0.e0, noises, self.__ft.size-highs)
    self.__yp = np.fft.irfft(self.__ft)

    
  def _smooth_signal(self):
    rft = np.copy(self.__ft)
    truncate = rft.size - int(0.01*self.__ntrnc * rft.size)
    rft[truncate:] = 0
    self.__ys = np.fft.irfft(rft)

  def _make_plot(self):

    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.plot(self.__x,self.__yi,label="Full", color="b")
    ax1.scatter(self.__x,self.__yp,label="Padded", s=10, color="k")

    # Plot both power spectra
    ax2.plot(self.__x,self.__yi, color="b")
    ax2.scatter(self.__x,self.__ys, s=10, color="k")

    # Add any plot frills you want
    ax1.set_ylim((-0.5,1.5))
    ax2.set_ylim((-0.5,1.5))
    ax1.set_xlabel("domain")
    ax1.set_ylabel("data")
    ax2.set_xlabel("domain")
    ax2.set_ylabel("data")
    ax1.legend(bbox_to_anchor=(1.05,1), loc="upper left", frameon=False)
    fig.tight_layout()
    return fig

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch_window()
  submodule.read_window()
