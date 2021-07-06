"""
Author: Ezra S. Brooker
Date Created: 2021 July 02
Date Modified: 

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

import tempfile, os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import PySimpleGUI as sg
import numpy as np

from sympy import var as symvar
from sympy import sympify
from sympy.utilities.lambdify import lambdify

from iea.utils.base_window import BaseWindow


class SubModuleWindow(BaseWindow):

  title = "FFTs: Digital Filtering"

  _figure_agg = None

  _xvar  = symvar("x")
  _fstr  = "x*cos(x**2)"
  _nsamp = 16
  _ntrnc = 0
  _x     = np.linspace(0,1,_nsamp)
  _yi    = _x*np.cos(_x**2)
  _ys    = _x*np.cos(_x**2)
  _yp    = _x*np.cos(_x**2)
  _ft    = np.fft.rfft(_yp)
  _snr   = 0.01


  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):

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
    snr_d = { "default_value":self._snr,   "enable_events":True, "key":"-SNR-",   "size":(4,5) }
    smp_d = { "default_value":self._nsamp, "enable_events":True, "key":"-SAMP-",  "size":(4,5) }
    trn_d = { "default_value":self._ntrnc, "enable_events":True, "key":"-TRUNC-", "size":(4,5) }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }
    buttons = [
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Text("<ENTER> a function --> f(x)", **txt_d)                         ],
      [ sg.InputText(self._fstr, **fun_d)                                      ],
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

    canvas = [[sg.Image(key="-IMAGE-")]]

    self.layout = [
        [
          sg.Col(buttons),
          sg.VSeperator(),
          sg.Col(canvas)
        ]
    ]

  def launch(self):
      self._generate_signal()
      self._smooth_signal()
      super().launch()
      self._draw()

  def event_loop(self):
      while True:
          event, values = self.window.read()
          close_now = self.check_event(event,values)
          if close_now:
            break

  def check_event(self,event,values):

    self._nsamp = int(values["-SAMP-"])
    self._ntrnc = int(values["-TRUNC-"])

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event == "-SUBMIT-":
      if self._fstr != values["-IFUNC-"]:
        self._fstr = values["-IFUNC-"]
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
      self._x = np.linspace(0,1,self._nsamp)
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
    self.window["-IMAGE-"].update(filename=self._fout.name)
    # if self._figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
    #     self._figure_agg.get_tk_widget().forget()
    #     plt.close("all")
    # figure_canvas_agg = FigureCanvasTkAgg(figure, self.window["-CANVAS-"].TKCanvas)
    # figure_canvas_agg.draw()
    # figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    # self._figure_agg =  figure_canvas_agg

  def _generate_signal(self):
    try:
      self._f = sympify(self._fstr)
      f = lambdify(self._xvar,self._f,"numpy")
    except:
      sg.Popup("[Error] Invalid Function!")
    self._yi = f(self._x)
    self._ys = f(self._x)
    self._yp = f(self._x)

  def _random_noise(self):
    signal     = np.mean(self._yi**2)
    sigavg     = 10*np.log10(signal)
    noises     = np.sqrt(10**((sigavg - self._snr)*0.1))
    self._yp  = self._yi + np.random.normal(0.e0, noises, self._nsamp)
    self._ft  = np.fft.rfft(self._yp)

  def _systematic_noise(self):
    self._random_noise()
    signal = np.mean(self._ft**2)
    sigavg = 10*np.log10(signal)
    noises = np.sqrt(10**((sigavg - self._snr*0.1)*0.1))
    highs  = int(0.85*self._ft.size)
    self._ft[highs:] += np.abs(np.random.normal(0.e0, noises, self._ft.size-highs))
    self._yp = np.fft.irfft(self._ft)

  def _smooth_signal(self):
    rft = np.copy(self._ft)
    truncate = rft.size - int(0.01*self._ntrnc * rft.size)
    rft[truncate:] = 0
    self._ys = np.fft.irfft(rft)

  def _make_plot(self):

    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.plot(self._x,self._yi,label="Full", color="b")
    ax1.scatter(self._x,self._yp,label="Padded", s=10, color="k")

    # Plot both power spectra
    ax2.plot(self._x,self._yi, color="b")
    ax2.scatter(self._x,self._ys, s=10, color="k")

    # Add any plot frills you want
    ax1.set_ylim((-0.5,1.5))
    ax2.set_ylim((-0.5,1.5))
    ax1.set_xlabel("domain")
    ax1.set_ylabel("data")
    ax2.set_xlabel("domain")
    ax2.set_ylabel("data")
    ax1.legend(bbox_to_anchor=(1.05,1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(self._fout.name)
    return fig

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()
