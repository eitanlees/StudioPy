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

  title="PDEs: Heat Equation"
  _solver = "Explicit FTCS"
  _npts = 64
  _func = lambda self,x: np.exp(-0.5*( (x-0.5)/0.05 )**2.0 )
  _BC_type = "Dirichlet"
  _levels = 10
  _norm = np.inf
  _cnt = 0
  _kappa = 0.2
  _dt = 1e-3
  _nsteps = 100

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    samp  = [16,32,64]
    slvs  = ["Explicit FTCS", "Backward Euler", "Crank Nicholson"]
    smp_d = { "default_value":self._npts, "enable_events":True, "key":"-PNTS-",  "size":(4,5) }
    slv_d = { "default_value":"Explicit FTCS", "enable_events":True, "key":"-SLVS-",  "size":(4,5) }
    stp_d = { "default_value":self._nsteps, "enable_events":True, "key":"-STEPS-",  "size":(35,20),  "resolution":10,  "orientation":"horizontal" }
    dif_d = { "default_value":self._kappa, "enable_events":True, "key":"-KAPPA-",  "size":(35,20), "resolution":0.01,  "orientation":"horizontal" }
    dts_d = { "size":(10,1), "enable_events": True, "key":"-DT-"}

    txt_d = { "size":(12,1) }
    nxt_d = { "size":( 10,1), "key":"-NEXT-"  }
    exi_d = { "size":( 10,1), "key":"-EXIT-"  }
    res_d = { "size":( 10,1), "key":"-RESET-"  }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }
    stb_d = { "size":(25,1), "key":"-STABLE-"}

    Radio_ExpFTCS = sg.Radio("Explicit FTCS", "-RADIO_SOLVER-", default=True, enable_events=True, key="-Explicit FTCS-")
    Radio_BackEul = sg.Radio("Backward Euler", "-RADIO_SOLVER-", default=False, enable_events=True, key="-Backward Euler-")
    Radio_CrankNich = sg.Radio("Crank Nicholson", "-RADIO_SOLVER-", default=False, enable_events=True, key="-CrankNich-")
    buttons = [
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Button("Steps - 0",  **nxt_d)                                    ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Combo(samp, **smp_d), sg.Text("# of Mesh Points",   size=(15,1))  ],
      [ sg.Slider((10,200), **stp_d), sg.Text("# of Steps Taken",   size=(15,1))  ],
      [ sg.Slider((0.0,2.0), **dif_d), sg.Text("Diffusion Constant", size=(15,1))  ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Input(self._dt, **dts_d), sg.Text("Timestep", size=(15,1))      ],
      [ sg.Text(f"Stable timestep <= {self._dt:1.2e}", **stb_d)                         ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Text("Solver Method", size=(12,1))                              ],
      [ Radio_ExpFTCS, Radio_BackEul, Radio_CrankNich                      ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Button("Exit", **exi_d), sg.Button("Reset", **res_d), sg.Button("Submit", **sub_d),          ],
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
      self._grid()
      self._ICs()
      super().launch()
      self._stable_dt()
      self._draw()

  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event in ("-STEPS-"):
      self._nsteps = int(values["-STEPS-"])

    elif event in ("-KAPPA-"):
      self._kappa = float(values["-KAPPA-"])
      self._stable_dt()

    elif event in ("-DT-"):
      self._dt = float(values["-DT-"])
      self._stable_dt()

    elif event in ("-RESET-", "-PNTS-", "-Explicit FTCS-", "-Backward Euler-", "-CrankNich-"):
      self._npts   = int(values["-PNTS-"])
      if values['-Explicit FTCS-']:
        self._solver = "Explicit FTCS"
      elif values['-Backward Euler-']:
        self._solver = "Backward Euler"
      elif values['-CrankNich-']:
        self._solver = "Crank Nicholson"
      
      self._grid()
      self._stable_dt()
      self._ICs()
      self._draw()

    elif event == "-NEXT-":
      if self._norm > 1e-6:
        self._cnt+=self._nsteps
        [self._heat_step() for _ in range(self._nsteps)]
        self._draw()
        self.window["-NEXT-"].update(f"Steps - {self._cnt}")


  def _draw(self):
    # Plot both datasets
    self.window["-NEXT-"].update(f"Steps - {self._cnt}")
    plt.close("all")
    
    fig, (ax1) = plt.subplots(1)

    ax1.scatter(self._x, self._u)
    
    # Add any plot frills you want
    if (not self._u.max() > self._vmax or not self._u.min() < self._vmin):
      ax1.set_ylim((self._vmin, self._vmax))
    ax1.set_xlim((0.0,1.0))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _grid(self):
    self._x = np.linspace(0.0,1.0,self._npts+1)
    self._h = self._x[1] - self._x[0]

  def _ICs(self):
    self._cnt = 0
    self._norm = np.inf
    self._u = self._func(self._x)
    self._BCs()
    self._u0 = self._u.copy()
    self._vmin = 0.0
    self._vmax = 1.25

  def _BCs(self):
    if self._BC_type == 'Dirichlet':
      self._u[0] = 0.0 #np.sin(np.sqrt(np.pi)*self._x[0]**2.0)
      self._u[-1] = 0.0 #np.exp(self._x[-1])

  def _stable_dt(self):
    self._stable = 0.5 * self._h**2 /self._kappa**2
    self.window["-STABLE-"].update(f"Stable timestep <= {self._stable:1.2e}")


  def _heat_step(self):
    if self._solver == "Explicit FTCS":
      self._heat_explicit()
    else:
      pass
    self._BCs()
    self._norm = la.norm(self._u-self._u0) / la.norm(self._u0)
    self._u0 = self._u.copy()

  def _heat_explicit(self):
    fact = self._dt * self._kappa**2 / self._h**2
    self._u[1:-1] += fact*(self._u[2:] - 2.0*self._u[1:-1] + self._u[:-2])
    
if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

