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

  title="Unconstrained Optimization in 2D"

  _funcstr0  = "100*(y-x**2)**2+(1-x)**2"
  _funcstr   = "100*(y-x**2)**2+(1-x)**2"
  _user_func = True

  _x = np.linspace(-np.pi,np.pi,500)
  _y = np.sin(_x)

  _norm = np.inf

  _alpha0 = 1.e0
  _gamma  = 0.5e0
  _rho    = 0.5e0

  _xin = -3.0
  _yin = -4.0
  _xmin,_xmax = -5.0,5.0
  _ymin,_ymax = -5.0,5.0

  _mode = "Newton"
  _backtracking_on = False

  _solvers = ['Newton', 'Steepest Descent', 'BFGS']

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch, makeTempFile=True)


  def _configure_layout(self):
    # sg.Theme("")
    buttons = [
      [ sg.Text("x-axis min and max  ", size=(18,1), key='-TEXT1-'), sg.In(self._xmin,size=(5,1),key='-XMIN-'), sg.In(self._xmax,size=(5,1),key='-XMAX-') ],
      [ sg.Text("y-axis min and max  ", size=(18,1), key='-TEXT2-'), sg.In(self._ymin,size=(5,1),key='-YMIN-'), sg.In(self._ymax,size=(5,1),key='-YMAX-') ],
      [ sg.Text("Input x,y coordinate", size=(18,1), key='-TEXT3-'), sg.In(self._xin,size=(5,1),key='-XIN-'), sg.In(self._yin,size=(5,1),key='-YIN-') ],
      [ sg.Checkbox("Use backtracking", size=(15,1), key='-BACKTRACK-'), sg.Combo(self._solvers, default_value=self._mode, size=(15,1), key='-SOLVERS-')],
      [ sg.Text("Initialized", key='-STATUS-', size=(25,1))], 
      [sg.Text(f"Stepsize = {self._norm}", key='-ERROR-', size=(25,1))],
      [ sg.Text(f"(x,y) = ({self._xin:1.2e},{self._yin:1.2e})", key='-SOLUTION-', size=(25,1))],
      [ sg.Text()],
      [ sg.Button("Solve", size=(8,3), key='-SOLV-'), sg.Button("Refresh", size=(8,3), key='-REFRESH-'), sg.Button("Exit", size=(8,3), key="-EXIT-") ],
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
 
    elif event == "-SOLV-":
      self.window['-STATUS-'].update(f"Solving: Step {self._iter}")
      while self._norm > 1e-6 and self._iter < 100:
        self._iterate()
        self._draw()
        self.window['-STATUS-'].update(f"Solving: Step {self._iter}")
        self.window['-ERROR-'].update(f"Stepsize = {self._norm:1.2e}")
        self.window['-SOLUTION-'].update(f"(x,y) = ({self._xc[-1]:1.2e},{self._yc[-1]:1.2e})")
      self.window['-STATUS-'].update(f"Solved: Step {self._iter}")
      
      


    elif event == "-REFRESH-":
      self.window['-STATUS-'].update("Refreshing...")
      self._xin = float(values['-XIN-'])
      self._yin = float(values['-YIN-'])
      self._xmin, self._xmax = float(values['-XMIN-']), float(values['-XMAX-'])
      self._ymin, self._ymax = float(values['-YMIN-']), float(values['-YMAX-'])
      self._backtracking_on = values['-BACKTRACK-']
      self._mode = values['-SOLVERS-']
      self._init()
      self._draw()
      self.window['-SOLUTION-'].update(f"(x,y) = ({self._xin:1.2e},{self._yin:1.2e})")
      self.window['-ERROR-'].update(f"Stepsize = {self._norm}")
      self.window['-STATUS-'].update("Initialized")


  def _draw(self):

    # Add any plot frills you want
    plt.close('all')
    plt.contour(self._X,self._Y,self._Z, levels=np.linspace(self._Z.min(),self._Z.max(),25), colors='w')
    plt.scatter(self._xc[-1],self._yc[-1], marker='*', c='y', s=100, zorder=100)
    plt.plot(self._xc, self._yc, 'o-', color='r', markersize=4)
    plt.xlabel("x")
    plt.ylabel("y")
    s = ""
    if self._backtracking_on: s+=" w/ backtracking"
    plt.title(self._mode+s)
    plt.xlim((self._xmin,self._xmax))
    plt.ylim((self._ymin,self._ymax))
    plt.tight_layout()
    plt.savefig(self._fout.name)
    self.window["-IMAGE-"].update(filename=self._fout.name)


  def _init(self):

    self._iter = 0
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
    self._v = np.array([[self._xin],[self._yin]])
    self._xc = [self._xin]
    self._yc = [self._yin]
    self._x = np.linspace(self._xmin,self._xmax,1000)
    self._y = np.linspace(self._ymin,self._ymax,1000)
    self._X, self._Y = np.meshgrid(self._x,self._y)
    self._Z = self._func(self._X,self._Y)
    self._norm = np.inf
    self._iter = 0

    if self._mode == "BFGS":
      self._B = np.identity(2)
      self._invB = np.identity(2)


  def _iterate(self):

    if self._mode == "Newton":
      self._newton()
    elif self._mode == "Steepest Descent":
      self._steepest_descent()
    elif self._mode == "BFGS":
      self._BFGS()

    else:
      self._newton()

    if self._backtracking_on:
      self._backtrack()
      self._p *= self._alpha
    self._norm = la.norm(self._p)
    if self._mode == "BFGS":
      self._vold = self._v.copy()
    self._v += self._p
    self._xc.append(self._v.squeeze()[0])
    self._yc.append(self._v.squeeze()[1])
    self._iter+=1


  def _backtrack(self):

    self._alpha = self._alpha0
    r1 = self._v + self._alpha*self._p
    f1 = self._func(*r1.squeeze()).astype(np.float64)
    f2a = self._func(*self._v.squeeze()).astype(np.float64)
    f2b = self._grad(*self._v.squeeze()).astype(np.float64).T
    f2b = f2b.squeeze()
    f2bp = np.dot(f2b,self._p.squeeze())
    f2 = f2a + self._gamma*self._alpha * f2bp

    liter = 0
    while f1 > f2 and liter < 1000:
      self._alpha *= self._rho
      r1 = self._v + self._alpha*self._p
      f1 = self._func(*r1.squeeze()).astype(np.float64)
      f2 = f2a + self._gamma*self._alpha * f2bp
      liter+=1


  def _newton(self):
    self._f = self._func(*self._v.squeeze()).astype(np.float64)
    self._g = self._grad(*self._v.squeeze()).astype(np.float64).T
    self._H = self._hess(*self._v.squeeze()).astype(np.float64)
    self._p = -1.e0 * la.solve(self._H, self._g)


  def _steepest_descent(self):
    self._p = -1.e0 * self._grad(*self._v.squeeze()).astype(np.float64).T


  def _BFGS(self):
    g    = self._grad(*self._v.squeeze()).astype(np.float64).T
    if self._iter > 0:
      gold = self._grad(*self._vold.squeeze()).astype(np.float64).T
      u    = g - gold
      s    = self._v - self._vold
      v    = np.matmul(self._B,s)
      alph =  1.e0 / np.dot(u.squeeze(),s.squeeze())
      beta = -1.e0 / np.dot(s.squeeze(),v.squeeze())

      uuT  = np.dot(u,u.T)
      vvT  = np.dot(v,v.T)
      self._B += alph*uuT + beta*vvT
      self._sherman_morrison(u,v,s)
    self._p = -1.e0 * np.matmul(self._invB,g)


  def _sherman_morrison(self,u,v,s):
    # embed()
    sTu = np.dot(s.T.squeeze(),u.squeeze())
    uBu = np.dot(np.matmul(u.T,self._invB).squeeze(),u.T.squeeze())
    ssT = np.dot(s,s.T)
    add = (sTu + uBu)*ssT / (sTu**2)

    BusT = np.matmul(np.matmul(self._invB,u), s.T)
    suTB = np.matmul(np.matmul(s,u.T), self._invB)
    sub  = (BusT + suTB) / sTu
    
    self._invB += add - sub
    

if __name__ == "__main__":

  submodule = SubModuleWindow()
  submodule.launch()
  submodule.event_loop()

