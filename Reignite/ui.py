#!/usr/bin/env python
import os
import sys

from manager import Manager
from webserver import WebServer

#
# entry point
#
if __name__ == "__main__":

  port = int(sys.argv[1])
  output_dir = sys.argv[2]
  # if len(sys.argv) >= 2:

  #   port = sys.argv[2]

  manager = Manager(output_dir)
  manager.start()

  webserver = WebServer(manager, port)
  webserver.start()
