"""
Ezra S. Brooker

    Department of Scientific Computing
    Florida State University


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

from sympy.abc import x,y
from sympy import ordered, Matrix, hessian, sympify, lambdify

from iea.utils.base_window import BaseWindow

from IPython import embed

plt.style.use("dark_background")
pi = np.pi

class SubModuleWindow(BaseWindow):

  title="SubModuleWindow Template"

  _funcstr0  = "100*(y-x**2)**2+(1-x)**2"
  _funcstr   = "100*(y-x**2)**2+(1-x)**2"
  _user_func = True

  _x = np.linspace(-np.pi,np.pi,500)
  _y = np.sin(_x)

  _norm = np.inf

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    buttons = [
      [ sg.Text("This is sg.Text ELEMENT", size=(20,1), key='-TEXT1-') ],
      [ sg.Button("Next", size=(10,3), key='-NEXT-') ],
      [ sg.Button("Refresh", size=(10,3), key='-REFRESH-') ],
      [ sg.Text("", size=(1,1))], # Blank line creator
      [ sg.Button("Exit", size=(10,3), key="-EXIT-") ],
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
      self._init()
      super().launch()
      self._draw()


  def event_loop(self):
      super().event_loop()


  def check_event(self, event, values):

    if event in self._EXIT_LIST + ["-EXIT-"]:
      self.window.close()
      return True
 
    elif event == "-NEXT-" and self._norm > 1e-6:
      self._iterate()
      self._draw()

    elif event == "-REFRESH-":
      self._init()
      self._draw()

  def _draw(self):

    # Add any plot frills you want

    plt.contour(self._X,self._Y,self._Z, levels=np.linspace(self._Z.min(),self._Z.max(),25), colors='w')
    embed()
    plt.scatter(*self._v)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Example Plot")
    plt.ylim((-2.5,2.0))
    plt.xlim((-2.5,2.0))
    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _init(self):

    if self._user_func:
      try:
        eqn        = sympify(self._funcstr)
        v          = list(ordered(eqn.free_symbols))
        gradient   = lambda f, v: Matrix([f]).jacobian(v)
        self._func = lambdify(v, eqn,             "numpy")
        self._grad = lambdify(v, gradient(eqn,v), "numpy")
        self._hess = lambdify(v, hessian(eqn,v),  "numpy")
      except:
        sg.Popup("[Error] Invalid function given!")
        self.window["-INPUT-FUNC-"].update(self._funcstr0)    
        eqn        = sympify(self._funcstr0)
        v          = list(ordered(eqn.free_symbols))
        gradient   = lambda f, v: Matrix([f]).jacobian(v)
        self._func = lambdify(v, eqn,             "numpy")
        self._grad = lambdify(v, gradient(eqn,v), "numpy")
        self._hess = lambdify(v, hessian(eqn,v),  "numpy")

    self._user_func = False
    self._v = np.array([0.0,1.0])
    self._x = np.linspace(-2.5,2.0,1000)
    self._y = np.linspace(-2.5,2.0,1000)
    self._X, self._Y = np.meshgrid(self._x,self._y)
    self._Z = self._func(self._X,self._Y)


  def _iterate(self):

    self._f = self._func(*self._v).astype(np.float64)
    self._g = -1.e0 * self._grad(*self._v).astype(np.float64)
    self._H = self._hess(*self._v).astype(np.float64)
    self._s = la.solve(self._H, self._g.T).squeeze()
    self._norm = la.norm(self._s)
    self._v += self._s


if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

