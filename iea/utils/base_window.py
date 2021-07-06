"""
Ezra S. Brooker
Date Created: 2021-07-05

Department of Scientific Computing
Florida State University

Base window class that will be used in all successive window classes.

It will mostly be used to mitigate the need for excessively rewriting
the configfile loading function and make dev-testing easier before
writing any necessary custom implementations of window class methods.

"""

import sys
if sys.version[0] != "3":
    raise Exception("Python 2 is no longer supported, please use Python 3")

import os, pathlib, yaml
import tempfile
import PySimpleGUI as sg


class BaseWindow:
  """ Base Window class for IEA package """

  _EXIT_LIST = [None, sg.WIN_CLOSED, "\x1b"]
  title = "Base Window"

  # Default window configuration
  wincfg = {
    "location": [100,100],
    "finalize": True,
    "resizable": True,
    "return_keyboard_events": True
  }


  def __init__(self, configfile=None, launch=False, makeTempFile=False):

    self._arch = self._determine_arch()

    if makeTempFile:
      tempdict = {"suffix": ".png"}
      if self._arch == "windows":
        tempdict['delete'] = False
      self._fout = tempfile.NamedTemporaryFile(**tempdict)

    self.configfile = configfile
    self._configure()
    if launch:
      self.launch()


  def __del__(self):
    if self._arch == "windows":
      self._fout.close()
      print(f"Removing Tempfile {self._fout.name}")
      os.remove(self._fout.name)


  def _determine_arch(self):
    # Used to determine OS architecture necessary for some submodules
    if sg.running_linux():
        return "linux"
    if sg.running_mac():
        return "mac"
    if sg.running_windows():
        return "windows"
    answer = input("Neither Linux/Mac/Windows detected, \n\
      IEA may not function correctly. \n\
      Wish to proceed [y]/N?")
    if answer.lower().replace(" ", "") in ("n", "no"):
      sys.exit("\nAborting IEA\n")

  def _configure(self):
    self._configure_window()
    self._configure_layout()


  def _configure_window(self):
    """ Configure the window layout and optionally load any configuration elements from a config file """

    # If valid config file provided, load window configuration data
    if self.configfile is not None:
      if os.path.isfile(self.configfile):
        extension = pathlib.Path(self.configfile).suffix.strip('.')
      else:
        raise Exception(f"[WindowConfigure Error] Configuration file does not exist: {self.configfile}")

      with open(self.configfile, "r") as f:
        if extension in ("yaml", "yml"):
          try:
            self.config = yaml.safe_load(f.read())
          except yaml.YAMLError as exc:
            print(exc)
        else:
          raise Exception(f"[WindowConfigure Error] Can't read configuration file extension: {self.configfile}")

      self.wincfg = self.config['window config']


  def _configure_layout(self):

    # Begin window layout configuration
    buttons = [
      [ sg.Button("Exit", key="-EXIT-") ],
      [ sg.Text(f"Test window for {self.title}")]
    ]

    # Finalize the window layout
    self.layout = [
      [sg.Col(buttons),]
    ]


  def launch(self):
      """ Launch the window, optionally begin immediate read (mostly for dev testing) """
      self.window = sg.Window(self.title, self.layout, **self.wincfg)


  def event_loop(self):
      """ Begin window reading event loop """
      while True:
          event, values = self.window.read()
          closed = self.check_event(event,values)
          if closed:
            break


  def check_event(self,event,values):
    """ Check event from event loop, this will be customized in each specific window implementation """
    if event in self._EXIT_LIST + ["-EXIT-"]:
        self.window.close()
        return True


if __name__ == "__main__":

  # Dev testing
  sub = BaseWindow()
  sub.launch()
  sub.event_loop()
