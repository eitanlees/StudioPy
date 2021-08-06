"""
Ezra S. Brooker
Date Created: 2021-07-05

    Department of Scientific Computing
    Florida State University

Poisson Equation Demonstration

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
import scipy.linalg as sla

from iea.utils.base_window import BaseWindow

plt.style.use("dark_background")
pi = np.pi
G = 6.674e-1
m = 5.0

class SubModuleWindow(BaseWindow):

  title="PDEs: Poisson Equation"
  _solver = "Jacobi"
  _npts = 16
  _func = lambda self,x: -np.random.random(x.size)
  _BC_type = "Dirichlet"
  _levels = 10
  _norm = np.inf
  _cnt = 0
  _w = 1.25
  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    samp  = [16,32,64]
    slvs  = ["Jacobi", "Gauss-Seidel", "SOR"]
    smp_d = { "default_value":self._npts, "enable_events":True, "key":"-PNTS-",  "size":(4,5) }
    slv_d = { "default_value":"Jacobi", "enable_events":True, "key":"-SLVS-",  "size":(4,5) }
    txt_d = { "size":(12,1) }
    nxt_d = { "size":( 10,1), "key":"-NEXT-"  }
    exi_d = { "size":( 10,1), "key":"-EXIT-"  }
    fun_d = { "size":(31,1), "key":"-IFUNC-" }
    smp_d = { "default_value":self._npts, "enable_events":True,   "key":"-PNTS-",  "size":(4,5) }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }

    Radio_Jacobi = sg.Radio("Jacobi", "-RADIO_SOLVER-", default=True, enable_events=True, key="-JACOBI-")
    Radio_GaussSeidel = sg.Radio("Gauss-Seidel", "-RADIO_SOLVER-", default=False, enable_events=True, key="-GAUSS-SEIDEL-")
    Radio_SOR = sg.Radio("SOR", "-RADIO_SOLVER-", default=False, enable_events=True, key="-SOR-")
    buttons = [
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Button("Steps - 0",  **nxt_d)                                    ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Combo(samp, **smp_d), sg.Text("# of Mesh Points", size=(15,1))  ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Text("Iterative Solver", size=(12,1))                           ],
      [ Radio_Jacobi, Radio_GaussSeidel, Radio_SOR                         ],
      [ sg.Text("", size=(1,1))                                            ],
      [ sg.Button("Exit", **exi_d), sg.Button("Submit", **sub_d),          ],
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
      self._matrix_op()
      self._forcing()
      self._ICs()
      self._BCs()
      self._fft()
      super().launch()
      self._draw()


  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event in ("-PNTS-", "-JACOBI-", "-GAUSS-SEIDEL-", "-SOR-"):
      self._npts   = int(values["-PNTS-"])
      if values['-JACOBI-']:
        self._solver = "Jacobi"
      elif values['-GAUSS-SEIDEL-']:
        self._solver = "Gauss-Seidel"
      elif values['-SOR-']:
        self._solver = "SOR"
      self._grid()
      self._matrix_op()
      self._forcing()
      self._ICs()
      self._BCs()
      self._fft()
      self._draw()

    elif event == "-NEXT-":
      if self._norm > 1e-3:
        self._cnt+=10
        [self._poisson_step() for _ in range(10)]
        self._fft()
        self._draw()
        self.window["-NEXT-"].update(f"Steps - {self._cnt}")


  def _draw(self):
    # Plot both datasets
    self.window["-NEXT-"].update(f"Steps - {self._cnt}")
    plt.close("all")
    
    fig, (ax1,ax2) = plt.subplots(2)

    ax1.scatter(self._x, self._u/la.norm(self._u))

    # ax1.scatter(self._x,self._g/la.norm(self._g))
    # Add any plot frills you want
    ax1.set_ylim((self._vmin, self._vmax))
    # ax1.set_xlim((0.0,1.0))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")


    ax2.scatter(self._freq, self._spec)
    ax2.set_yscale("log",basey=2)
    ax2.set_ylim((2**(-4),2**8))
    ax2.set_xlim(left=0.0)
    ax2.set_xlabel("frequency")
    ax2.set_ylabel("power")

    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _grid(self):
    self._x = np.geomspace(5.0,0.001,self._npts)
    self._h = np.min(np.abs(np.diff(self._x)))

  def _ICs(self):

    self._g = - G * m / self._x
    self._g /= la.norm(self._g)

    self._cnt = 0
    self._norm = np.inf
    self._u = self._func(self._x)
    self._u /= la.norm(self._u)
    self._BCs()
    self._u0 = self._u.copy()
    self._vmin = -0.5
    self._vmax = 0.1

  def _BCs(self):
    if self._BC_type == 'Dirichlet':
      self._u[-1] = self._g.min()
      self._u[0] = 0.0

  def _matrix_op(self):

    # Create matrix A and set main diagonal
    temp = np.ones(self._npts)
    temp[1:-1] *= 2.0
    self._A =  np.diag(temp)

    # Set sub and super diagonals (k=-1,1)
    temp = -1*np.ones(self._npts-1)
    temp[-1] = 0.0
    self._A += np.diag(temp,-1)
    self._A += np.diag(temp[::-1],1)
    self._A /= self._h**2


  def _forcing(self):
    self._fx = np.zeros_like(self._x)
    self._fx = - G * m / self._x
    self._fx /= la.norm(self._fx)

  def _jacobi(self):
    for i in range(self._npts):
      self._u[i]  = self._fx[i]
      self._u[i] -= sum([ self._A[i,j]*self._u0[j] for j in range(self._npts) if j != i ])
      self._u[i] /= self._A[i,i]


  def _gauss_seidel(self):
    for i in range(self._npts):
      temp = sum([ self._A[i,j]*self._u[j] for j in range(self._npts) if j != i ])
      self._u[i] = (self._fx[i] - temp) / self._A[i,i]


  def _sor(self):
    for i in range(self._npts):

      left  = sum([self._A[i,j]*self._u[j] for j in range(i)])
      right = sum([self._A[i,j]*self._u0[j] for j in range(i+1,self._npts)])

      self._u[i] = (1.0-self._w)*self._u0[i] + self._w*(self._fx[i] - left - right)/self._A[i,i]

  def _poisson_step(self):
    if self._solver=='Jacobi':
      self._jacobi()
    elif self._solver == "Gauss-Seidel":
      self._gauss_seidel()
    elif self._solver == "SOR":
      self._sor()

    self._BCs()
    self._norm = la.norm(self._u-self._u0) / la.norm(self._u0)
    self._u0 = self._u.copy()



  def _fft(self):
      self._dft  = np.fft.fft(self._u)
      self._spec = np.abs(self._dft * np.conj(self._dft))
      self._freq = np.fft.fftfreq(self._npts, 1.0/self._npts)
      self._spec = np.where(self._spec < 2**(-4), 2**-4, self._spec)

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

