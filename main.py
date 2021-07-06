"""
Ezra S. Brooker
Date Created: 2021

Department of Scientific Computing
Florida State University

Main driver of the Interactive Educational Applet that is built
in Python based off of Michael Heath's Scientific Computing java
modules.

"""
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import os
import PySimpleGUI as sg

from iea.course.course_window import CourseWindow


def main():

  # Hardcoded to point to a specific course config, will abstract away, eventually
  # Sets up home page for course window
  course = CourseWindow(configfile='iea/acs2/config/acs2.yml', launch=True)

  # Event loop used to allow both home window and submodule windows to coexist
  while True:
    window, event, values = sg.read_all_windows()

    if window == course.window:
      # Check home window events
      close_all = course.check_event(event,values)
      if close_all:
        break
    else:
      # Check for submodule window event
      window_closed = course.subwindows[window].check_event(event,values)
      if window_closed:
        course.subwindows.pop(window)

  # [submod.window.close() for submod in course.subwindows.values()]
  course.window.close()

if __name__ == '__main__':

  main()
