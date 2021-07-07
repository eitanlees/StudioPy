"""
Ezra S. Brooker
Date Created: 2021-07-05

    Department of Scientific Computing
    Florida State University

Laplace Equation Demonstration

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

  title="PDEs: Laplace Equation"

  _npts = 20
  _func = lambda junk,x,y: y*np.sin(pi*x**2) + x*np.cos(pi*y**2)
  _BCs = "Dirichlet"
  _levels = 10
  _norm = np.inf
  _cnt = 0
  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")

    txt_d = { "size":(30,1) }
    nxt_d = { "size":( 10,1), "key":"-NEXT-"  }
    exi_d = { "size":( 10,1), "key":"-EXIT-"  }
    fun_d = { "size":(31,1), "key":"-IFUNC-" }
    trn_d = { "default_value":self._npts, "enable_events":True, "key":"-TRUNC-", "size":(4,5) }
    sub_d = { "bind_return_key":True,       "visible":False,      "key":"-SUBMIT-" }
    buttons = [
      [ sg.Text("", size=(1,1))                                                 ],
      [ sg.Button("Next - 0",  **nxt_d)                                             ],
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
      self._grid()
      self._ICs()
      self._BCs()
      super().launch()
      self._draw()


  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event == "-NEXT-":
      while self._norm > 1e-3:
        self._cnt+=1
        [self._laplace_step() for _ in range(10)]
        self._draw()
        self.window["-NEXT-"].update(f"Next - {self._cnt}")



  def _draw(self):
    # Plot both datasets
    plt.close("all")
    plt.contourf(self._Xm, self._Ym, self._u, levels=self._contours, cmap="inferno")

    # Add any plot frills you want
    plt.ylim((0.0,1.0))
    plt.xlim((0.0,1.0))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _grid(self):
    self._x = np.linspace(0.0,1.0,self._npts)
    self._y = self._x.copy()
    self._Xm, self._Ym = np.meshgrid(self._x, self._y)


  def _ICs(self):
    self._u = np.zeros((self._npts, self._npts))
    for i in range(self._npts):
      self._u[i,:] = self._func(self._x[i],self._y)
    self._BCs()
    self._u0 = self._u.copy()
    self._contours = np.linspace(0.9*self._u.min(),1.1*self._u0.max(), self._levels)


  def _laplace_step(self):
    for i in range(1,self._npts-1):
      for j in range(1,self._npts-1):
        self._u[i,j] = 0.25 * ( self._u0[i+1,j] +\
                                self._u0[i-1,j] +\
                                self._u0[i,j+1] +\
                                self._u0[i,j-1] )
    self._BCs()
    self._norm = la.norm(self._u-self._u0) / la.norm(self._u0)
    self._u0 = self._u.copy()


  def _BCs(self):
    if self._BCs == 'Dirichlet':
      self._u[0,:] = np.cos(0.5*pi*(x**2-y**2))
      self._u[-1,:] = 0.5
      self._u[:,0] = 1.0
      self._u[:,-1] = np.sin(1.5*pi*x)

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

