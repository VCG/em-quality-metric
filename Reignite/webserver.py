import json
import mahotas as mh
import numpy as np
import os
import socket
import time
import tornado
import tornado.gen
import tornado.web
import tornado.websocket
import urllib
import StringIO

from uitools import UITools

from PIL import Image


class WebServerHandler(tornado.web.RequestHandler):

  def initialize(self, webserver):
    self._webserver = webserver

  @tornado.web.asynchronous
  @tornado.gen.coroutine
  def get(self, uri):
    '''
    '''
    self._webserver.handle(self)


class WebServer:

  def __init__( self, manager, port=2001 ):
    '''
    '''
    self._manager = manager
    self._port = port

    self._image_function_cache = []

  def start( self ):
    '''
    '''

    ip = socket.gethostbyname('')
    port = self._port

    webapp = tornado.web.Application([
      
      # (r'/tree/(.*)', WebServerHandler, dict(webserver=self)),
      # (r'/type/(.*)', WebServerHandler, dict(webserver=self)),
      # (r'/content/(.*)', WebServerHandler, dict(webserver=self)),
      # (r'/metainfo/(.*)', WebServerHandler, dict(webserver=self)),
      # (r'/data/(.*)', WebServerHandler, dict(webserver=self)),
      # (r'/query/(.*)', WebServerHandler, dict(webserver=self)),
      (r'/merge/(.*)', WebServerHandler, dict(webserver=self)),
      (r'/split/(.*)', WebServerHandler, dict(webserver=self)),
      (r'/(.*)', tornado.web.StaticFileHandler, dict(path=os.path.join(os.path.dirname(__file__),'web'), default_filename='index.html'))
  
    ])

    webapp.listen(port, max_buffer_size=1024*1024*150000)

    print 'Starting webserver at \033[93mhttp://' + ip + ':' + str(port) + '\033[0m'

    tornado.ioloop.IOLoop.instance().start()

  @tornado.gen.coroutine
  def handle( self, handler ):
    '''
    '''
    content = None

    splitted_request = handler.request.uri.split('/')

    #
    # store
    #
    if splitted_request[2] == 'store':
      self._manager.store()


    if splitted_request[1] == 'merge':

      image_function = self._manager.get_merge_error_image
      correction_function = self._manager.correct_merge
      error = self._manager.get_next_merge_error()
      correction_count = len(error[3])

    elif splitted_request[1] == 'split':

      image_function = self._manager.get_split_error_image
      correction_function = self._manager.correct_split
      error = self._manager.get_next_split_error()
      correction_count = 1


    if splitted_request[2] == 'get_correction_count':

      self._image_function_cache = []
      for i in range(correction_count):
        border_before, border_after, labels_before, labels_after, slice_overview = image_function(error, i)
        self._image_function_cache.append([border_before, border_after, labels_before, labels_after, slice_overview])
    
      content = json.dumps({'correction_count': correction_count})
      content_type = 'text/html'

    elif splitted_request[2] == 'correct':

      clicked_correction = splitted_request[3];
      new_mode = correction_function(clicked_correction);

      content = json.dumps({'mode':new_mode})
      content_type = 'text/html'



        

    if not content:

      image = np.zeros((1,1))
      number = int(splitted_request[3])
      border_before, border_after, labels_before, labels_after, slice_overview = self._image_function_cache[number]

      if splitted_request[2] == 'slice_overview':

        image = slice_overview

      elif splitted_request[2] == 'current':

        image = border_before

      elif splitted_request[2] == 'current_labels':

        image = labels_before

      elif splitted_request[2] == 'correction':

        image = border_after

      elif splitted_request[2] == 'correction_labels':
        
        image = labels_after

      if image.shape[0]!=1:

        image = Image.fromarray(image, 'RGBA')
        output = StringIO.StringIO()
        image.save(output, 'PNG')

        content_type = 'image/png'
        content = output.getvalue()



    # invalid request
    if not content:
      content = 'Error 404'
      content_type = 'text/html'

    # handler.set_header('Cache-Control','no-cache, no-store, must-revalidate')
    # handler.set_header('Pragma','no-cache')
    # handler.set_header('Expires','0')
    handler.set_header('Access-Control-Allow-Origin', '*')
    handler.set_header('Content-Type', content_type)
    handler.write(content)
