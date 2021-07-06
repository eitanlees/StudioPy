# DSC_IEA
Interactive Educational Applet for the Florida State University Department of Scientific Computing courses. Inspired by the interactive educational modules developed for Michael Heath's scientific computing textbook.

# Use
At this time, the interactive educational applet is best installed through `conda` to easily obtain the necessary package `python-graphviz` due to an annoying issue found on Windows 8 (not tested on Windows 10, yet). There are ways to get around this by installing Graphviz and dealing with PATH names appropriately, but using `conda` is simply easier.

To setup the environment correctly, you can create a new `conda` environment or install dependencies in an existing environment. There is a requirements.txt file provided with this repo to faciliate setup.

For a new environment, use `conda create --name ENV-NAME -c conda-forge --file requirements.txt`. It is required to specify the `conda-forge` channel to obtain PySimpleGUI through `conda`. In an existing `conda` environment, simply use `conda install --name ENV-NAME -c conda-forge --file requirements.txt`

Once these packages are installed, simply use `python main.py` in the top-level directory to run the applet and learn!

