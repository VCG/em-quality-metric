#!/usr/bin/env python
import os
import sys

from webserver import WebServer

CACHE = {}

#
# entry point
#
if __name__ == "__main__":

  port = 2001
  if len(sys.argv) == 2:
    port = sys.argv[1]

  # manager = _mbeam.Manager()
  # manager.start()

  webserver = WebServer(port)
  webserver.start()
