import json
import os
import socket
import time
import tornado
import tornado.gen
import tornado.web
import tornado.websocket
import urllib


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

  def __init__( self, port=2001 ):
    '''
    '''
    # self._manager = manager
    self._port = port

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

    path = '/'.join(splitted_request[2:])

    if splitted_request[1] == 'tree':

      data_path = path.split('?')[0]
      parameters = path.split('?')[1].split('&')
      
      if parameters[0][0] != '_':
        data_path = urllib.unquote(parameters[0].split('=')[1])
      else:
        data_path = None
      
      content = json.dumps(self._manager.get_tree(data_path))
      content_type = 'text/html'

    elif splitted_request[1] == 'type':

      content = self._manager.check_path_type(path)
      if not content:
        content = 'NULL'
      content_type = 'text/html'

    elif splitted_request[1] == 'content':

      content = json.dumps(self._manager.get_content(path))
      content_type = 'text/html'

    elif splitted_request[1] == 'metainfo':

      content = self._manager.get_meta_info(path)
      content_type = 'text/html'

    elif splitted_request[1] == 'query':

      path = '/'.join(splitted_request[2:-1])

      tile = splitted_request[-1].split('-')

      i = int(tile[0])
      j = int(tile[1])

      content = self._manager.get_query(path, i, j)
      content_type = 'text/html'

    elif splitted_request[1] == 'data':

      # this is for actual image data
      path = '/'.join(splitted_request[2:-1])

      tile = splitted_request[-1].split('-')

      x = int(tile[1])
      y = int(tile[2])
      z = int(tile[3])
      w = int(tile[0])

      content = self._manager.get_image(path, x, y, z, w)
      content_type = 'image/jpeg'



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
