import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")
import os
import PySimpleGUI as sg

from course_window import CourseWindow

from IPython import embed


def main():

  course = CourseWindow(configfile='config/acs2.yml',launch_window=True)

  while True:

    window, event, values = sg.read_all_windows()

    if window == course.window:
      close_all = course.check_read(event,values)
      if close_all:
        break
    else:
      window_closed = course.submods[window].check_read(event,values)
      if window_closed:
        course.submods.pop(window)

  [submod.window.close() for submod in course.submods.values()]
  course.window.close()

if __name__ == '__main__':

  main()
