"""
Author: Ezra S. Brooker
Date Created: 2021 June 18
Date Modified: 

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

  title = "FFTs: Padding Sequences"

  _figure_agg = None

  _choices = ["zeros","constant","linear2zero","linear2periodic"]
  _textchs = ["Zeros", "Constant", "Linear to Zero", "Linear Periodic"]
  _textdct = dict(zip(_textchs,_choices))

  _funcstr = "x*cos(x**2)"
  _padding = "Zeros"
  _nsamps  = 33
  _x       = symvar("x")


  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch)


  def _configure_layout(self):

    # Set up the course window BACK and EXIT buttons
    sg.theme("Dark")

    buttons = [
      [ sg.Button("Generate Random Data", key="-RAND-"), sg.Button("Exit", key="-EXIT-")                                                                               ],
      [ sg.Text("Function f(x): PRESS <ENTER>",   size=(25,1)), sg.InputText(self._funcstr, key="-INPUT-FUNC-", size=(20,1))                                          ],
      [ sg.Text("Number of points", size=(25,1)), sg.Combo([i for i in range(1,65)], key="-INPUT-NUM-", enable_events=True, default_value=self._nsamps, size=(20,20)) ],
      [ sg.Text("Padding Method",   size=(25,1)), sg.Combo(self._textchs, size=(20,20), key="-INPUT-PAD-", default_value=self._padding, enable_events=True)          ],
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

  def launch(self):
      self._data()
      self._pad()
      super().launch()
      self._draw()

  def event_loop(self):
      while True:
          event, values = self.window.read()
          closed = self.check_read(event,values)
          if closed:
            break

  def check_read(self,event,values):

    self._funcstr = values["-INPUT-FUNC-"]
    self._padding = self._textdct[values["-INPUT-PAD-"]]
    self._nsamps  = int(values["-INPUT-NUM-"]) - 1

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True

    elif event == "-SUBMIT-":
      self._data()
      self._pad()

    elif event in ("-INPUT-PAD-","-INPUT-NUM-"):
      self._pad()

    elif event == "-RAND-":
      self._f = lambda x: np.random.random(x.size)
      self._data(user_func=False)
      self._pad()

    self._draw()

    return False

  def _draw(self):
    figure = self._plot_fft()
    if self._figure_agg: # ** IMPORTANT ** Clean up previous drawing before drawing again
        self._figure_agg.get_tk_widget().forget()
        plt.close("all")
    figure_canvas_agg = FigureCanvasTkAgg(figure, self.window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    self._figure_agg =  figure_canvas_agg


  def _data(self, user_func=True):
    if user_func:
      try:
        expr      = sympify(self._funcstr)
        self._f  = lambdify(self._x,expr,"numpy")
      except:
        sg.Popup("[Error] Invalid function given!")
    self._xf = np.linspace(0,1,64)    # Spatial domain for data
    self._yf = self._f(self._xf)    # Create full dataset


  def _pad(self):
    # Now pad the truncated the sequence
    self._yp = self._yf.copy()
    nd = self._xf.size - self._nsamps

    if self._padding.lower() == "zeros":
      self._yp[self._nsamps:] = np.zeros(nd)

    elif self._padding.lower() == "constant":
      self._yp[self._nsamps:] = np.ones(nd)*self._yp[-1]
      
    elif self._padding.lower() == "linear2zero":
      self._yp[self._nsamps:] = np.interp(self._xf[self._nsamps:], 
                                            [self._xf[self._nsamps], self._xf[-1]], 
                                            [self._yp[self._nsamps], 0.0])
      
    elif self._padding.lower() == "linear2periodic":
      self._yp[self._nsamps:] = np.interp(self._xf[self._nsamps:], 
                                            [self._xf[self._nsamps], self._xf[-1]], 
                                            [self._yp[self._nsamps], self._yp[0]])


  def _plot_fft(self):

    # Compute FFT and power spectra of both sequences
    fps = np.abs(np.fft.fft(self._yf))**2 # original
    pps = np.abs(np.fft.fft(self._yp))**2 # padded

    fps[fps < 2**(-6)] = 2**-6
    pps[pps < 2**(-6)] = 2**-6

    # Compute FFT frequences and sort them
    freq = np.fft.fftfreq(self._yf.size,1.0/self._yf.size)
    idx = np.argsort(freq)

    # Plot both datasets
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.scatter(self._xf,self._yf,label="Full", s=10, color="b")
    ax1.scatter(self._xf,self._yp,label="Padded", s=10, color="k")

    # Plot both power spectra
    ax2.scatter(freq[idx],fps[idx], s=10, color="b")
    ax2.scatter(freq[idx],pps[idx], s=10, color="k")

    # Add any plot frills you want
    ax1.set_xlabel("spatial domain")
    ax1.set_xlim(left=0.0)
    ax1.set_ylabel("data")
    ax2.set_xlabel("frequency")
    ax2.set_xlim(left=0.0)
    ax2.xaxis.set_ticks(np.arange(0,self._yf.size//2+1,8))
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
  window.launch()
  window.event_loop()
