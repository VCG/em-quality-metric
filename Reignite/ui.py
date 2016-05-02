#!/usr/bin/env python
import os
import sys

from manager import Manager
from webserver import WebServer

CACHE = {}

#
# entry point
#
if __name__ == "__main__":

  port = 2001
  if len(sys.argv) == 2:
    port = sys.argv[1]

  manager = Manager()
  manager.start()

  webserver = WebServer(manager, port)
  webserver.start()
