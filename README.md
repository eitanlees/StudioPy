# StudioPy
Interactive Educational Applet for the Florida State University Department of Scientific Computing courses. Inspired by the interactive educational modules developed for Michael Heath's scientific computing textbook.


# Use
At this time, the interactive educational applet is best installed through `conda` to easily obtain the necessary package `python-graphviz` due to an annoying issue found on Windows 8 (not tested on Windows 10, yet). There are ways to get around this by installing Graphviz and dealing with PATH names appropriately, but using `conda` is the easier/safer option.

Suggested setup is via a fresh `conda` environment:

`conda create --name ENV-NAME -c conda-forge --file requirements.txt`

`conda activate ENV-NAME` or `source activate ENV-NAME` if using MacOS

Use `python main.py` in the parent level StudioPy directory.


# Development
The general infrastructure for the applet is in place. There will be modifications as needed, if needed, but for now focus is on educational content submodules. There will be documentation provided very soon on designing, implementing, integrating, and testing new submodules for the new applet.


# Bug Notes
There are some small bugs littered throughout that can cause crashes from unexpected user input. Those will be solved with some exception handling. For the most part, this won't affect use of the applet, however, they do need to be taken care of soon so that a systematic infrastructure can be put in place as this will ultimately be an issue for future submodules developed by anybody.

# Other Notes
Unit testing of some sort will be implemented soon, as well, improve the developer experience.
