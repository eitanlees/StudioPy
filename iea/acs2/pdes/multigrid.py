"""
Ezra S. Brooker

Department of Scientific Computing
Florida State University

2021-08-06

Submodule for PDEs to demonstrate the multigrid method of iterative solvers

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

ZERO, ONE, TWO, FOUR = 0.e0,1.e0,2.e0,4.e0
HALF, THIRD, FOURTH = 0.5e0, 1.e0/3.e0, 0.25e0

class SubModuleWindow(BaseWindow):

  title="Multigrid Solver"

  _n = 257
  _m = 129
  _w = TWO*THIRD
  _cnt = 0
  _nsteps = 1

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    exi_d = { "size":( 15,1), "key":"-EXIT-"  }
    nxt_d = { "size":( 15,1), "key":"-NEXT-"  }
    res_d = { "size":( 15,1), "key":"-RESET-"  }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }
    stp_d = { "size":(10,1), "enable_events": True, "key":"-STEPS-"}

    buttons = [
      [ sg.Text("", size=(20,1), key='-TEXT1-') ],
      [ sg.Input(self._nsteps, **stp_d), sg.Text("# of Steps Taken",   size=(15,1))  ],
      [ sg.Button("Steps - 0",  **nxt_d)],
      [ sg.Button("Reset", **res_d)],
      [ sg.Button("Exit", **exi_d), sg.Button("Submit", **sub_d)],
      ]

    canvas = [[sg.Image(key="-IMAGE-")]]

    self.layout = [
        [
          sg.Col(buttons),
          sg.VSeperator(),
          sg.Col(canvas),
        ]
    ]

  def launch(self):
      super().launch()
      self._init()
      self._fft()
      self._norm = np.inf
      self._draw()


  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True

    elif event in ("-STEPS-"):
      try:
        self._nsteps = int(values["-STEPS-"])
      except:
        self._nsteps = 1

    elif event == "-NEXT-":
      self._cnt+=self._nsteps
      if la.norm(self._ux-self._uj1) > 1e-4: [self._just_jacobi() for _ in range(self._nsteps)]
      if la.norm(self._ux-self._u)   > 1e-4: [self._multigrid()   for _ in range(self._nsteps)]
      self._fft()
      self._draw()
      self.window["-NEXT-"].update(f"Steps - {self._cnt}")

    elif event == "-RESET-":
      self._init()
      self._fft()
      self._draw()
      self._cnt=0
      self.window["-NEXT-"].update(f"Steps - {self._cnt}")      


  def _draw(self):

    # Add any plot frills you want
    plt.close('all')

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(self._x,self._uj1,     label='Jacobi')
    ax1.plot(self._x,self._u,       label='Multigrid')
    ax1.plot(self._x,self._ux, ':', label='Exact')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_ylim(-1,8)


    ax2.scatter(self._freq_wj, self._spec_wj, label='Jacobi', s=4.0)
    ax2.scatter(self._freq_mg, self._spec_mg, label='Multigrid', s=4.0)
    ax2.set_yscale("log",basey=2)
    ax2.set_ylim((2**(-12),2**12))
    ax2.set_xlim(left=0.0)
    ax2.set_xlabel("frequency")
    ax2.set_ylabel("power spectra of residuals")

    # plt.title("Example Plot")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _init(self):

    self._x  = np.linspace(0,1,self._n)
    self._fx = FOUR * np.exp(TWO*self._x)
    self._fv = np.reshape(self._fx, (self._fx.size,1))
    self._ux = np.exp(TWO*self._x)
    self._h  = self._x[1] - self._x[0]
    self._x2 = np.linspace(0,1,self._m)
    self._h2 = self._x2[1] - self._x2[0]
    self._bfx = self._fx * (self._h**TWO)
    self._u0 = np.zeros_like(self._x)
    self._u0[0]  = ONE
    self._u0[-1] = np.exp(TWO)
    self._u   = self._u0.copy()
    self._uj0 = self._u0.copy()
    self._uj1 = self._u0.copy()
    self._A   = np.diag(np.ones(self._n-1),k=-1) + np.diag(-TWO*np.ones(self._n), k=0) + np.diag(np.ones(self._n-1),k=1)
    self._A  /= self._h**TWO
    
    self._A2  = np.diag(np.ones(self._m-1),k=-1) + np.diag(-TWO*np.ones(self._m), k=0) + np.diag(np.ones(self._m-1),k=1)
    self._A2 /= self._h2**TWO


  def _just_jacobi(self):

    w = self._w
    self._uj1 = self._uj0.copy()
    for i in range(1,self._n-1):
      b = FOUR*np.exp(TWO*self._x[i]) *  self._h2**TWO
      self._uj1[i] = (HALF * w) * (self._uj0[i-1] + self._uj0[i+1] - b) + (ONE - w) * self._uj0[i]

    self._uj1[0]  = self._uj0[0]
    self._uj1[-1] = self._uj0[-1]
    self._uj0 = self._uj1.copy()


  def _restrict(self, coarse, fine):

    coarse[0]  = fine[0]
    coarse[-1] = fine[-1]

    for i in range(1,coarse.size-1):
      j = 2*i
      coarse[i] = FOURTH * (fine[j-2] + TWO*fine[j-1] + fine[j])

    return coarse


  def _prolong(self, coarse, fine):

    for i in range(fine.size-1):
      if i%2 == 0:
        fine[i] = coarse[(i+1)//2]
      else:
        fine[i] = HALF * (coarse[i//2] + coarse[i//2 + 1])

    return fine


  def _weighted_jacobi_step(self):

    w = self._w
    self._uj = self._u0.copy()
    for i in range(1,self._n-1):
      b = FOUR*np.exp(TWO*self._x[i]) *  self._h2**TWO
      self._uj[i] = (HALF * w) * (self._u0[i-1] + self._u0[i+1] - b) + (ONE - w) * self._u0[i]

    self._uj[0]  = self._u0[0]
    self._uj[-1] = self._u0[-1]
    self._u0 = self._uj.copy()


  def _multigrid(self):

    for _ in range(2):
      self._weighted_jacobi_step()
    self._u = self._u0.copy()

    self._uv = np.reshape(self._u, (self._u.size,1))
    self._fine = np.reshape(self._fx,(self._n,1)) - np.matmul(self._A,self._uv)
    self._fine = self._fine.flatten()
    self._fine[0], self._fine[-1] = ZERO, ZERO

    self._coarse = np.zeros(self._m)
    self._coarse = self._restrict(self._coarse, self._fine)
    self._coarse[0], self._coarse[-1] = ZERO, ZERO

    # self._coarse = self._thomas_solver(self._coarse)
    self._e2h = la.solve(self._A2, self._coarse)
    self._e2h[0], self._e2h[-1] = ZERO, ZERO

    self._eh = np.zeros_like(self._fine)
    self._eh = self._prolong(self._e2h, self._eh)
    self._eh[0], self._eh[-1] = ZERO, ZERO

    self._u += self._eh
    self._u[0],self._u[-1] = self._u0[0], self._u0[-1]
    self._norm = la.norm(self._u0-self._u)
    self._u0 = self._u.copy()

  def _fft(self):
      self._dft  = np.fft.fft(np.abs(self._ux - self._u))
      self._spec_mg = self._dft * np.conj(self._dft)
      self._freq_mg = np.fft.fftfreq(self._n, 1.0/(self._n))


      self._dft  = np.fft.fft(np.abs(self._ux - self._uj1))
      self._spec_wj = self._dft * np.conj(self._dft)
      self._freq_wj = np.fft.fftfreq(self._n, 1.0/(self._n))

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

