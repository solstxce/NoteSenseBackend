import http.server
import socketserver

PORT = 3005
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving test page at http://localhost:{PORT}")
    httpd.serve_forever() 