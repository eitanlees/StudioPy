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

  course = CourseWindow(configfile='iea/acs2/config/acs2.yml', launch=True)

  while True:
    window, event, values = sg.read_all_windows()

    if window == course.window:
      close_all = course.check_event(event,values)
      if close_all:
        break
    else:
      window_closed = course.subwindows[window].check_event(event,values)
      if window_closed:
        course.subwindows.pop(window)

  # [submod.window.close() for submod in course.subwindows.values()]
  course.window.close()

if __name__ == '__main__':

  main()
