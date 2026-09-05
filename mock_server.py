from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

# Global UI state: toggle between 'original' and 'redesigned'
current_ui_version = "original"

def get_html_content():
    if current_ui_version == "original":
        return """<!DOCTYPE html>
<html>
<head><title>Mock Bank Portal</title></head>
<body style="font-family: Arial; padding: 20px;">
  <h2>Mock Commercial Bank - Statements Portal</h2>
  <div id="auth-section">
    <input type="text" name="username" placeholder="Username" /><br><br>
    <input type="password" name="password" placeholder="Password" /><br><br>
    <button id="submit-btn" onclick="document.getElementById('auth-section').style.display='none'; document.getElementById('portal-section').style.display='block';">Sign In</button>
  </div>
  
  <div id="portal-section" style="display:none; margin-top:20px;">
    <h3>Account Overview</h3>
    <a id="statements-link" href="#" onclick="document.getElementById('stmt-list').style.display='block';">Statements</a>
    
    <div id="stmt-list" style="display:none; margin-top:15px;">
      <p>August 2026 Monthly Statement</p>
      <a class="download-statement" href="/download_pdf" download="statement.pdf">
        <button>Download Statement</button>
      </a>
    </div>
  </div>
</body>
</html>"""
    else:
        # Redesigned Layout: Changed IDs, renamed links, and restructured layout
        return """<!DOCTYPE html>
<html>
<head><title>Mock Bank Portal (Redesigned)</title></head>
<body style="font-family: Arial; padding: 20px;">
  <h2>Mock Commercial Bank - Statements Portal (v2.0)</h2>
  <div id="auth-section">
    <input type="text" name="username" placeholder="Username" /><br><br>
    <input type="password" name="password" placeholder="Password" /><br><br>
    <button id="auth-submit-v2" onclick="document.getElementById('auth-section').style.display='none'; document.getElementById('portal-section').style.display='block';">Log In</button>
  </div>
  
  <div id="portal-section" style="display:none; margin-top:20px;">
    <h3>Account Documents</h3>
    <!-- The link was renamed and ID removed -->
    <a class="nav-tab-docs" href="#" onclick="document.getElementById('stmt-list').style.display='block';">Documents & Tax Statements</a>
    
    <div id="stmt-list" style="display:none; margin-top:15px;">
      <p>August 2026 Monthly Statement</p>
      <a id="btn-export-pdf" href="/download_pdf" download="statement.pdf">
        <button>Export Document</button>
      </a>
    </div>
  </div>
</body>
</html>"""

class MockBankHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global current_ui_version
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(get_html_content().encode("utf-8"))
        elif self.path == "/download_pdf":
            self.send_response(200)
            self.send_header("Content-type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="statement.pdf"')
            self.end_headers()
            # Generate minimal valid dummy PDF bytes
            dummy_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000117 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n200\n%%EOF"
            self.wfile.write(dummy_pdf)
        elif self.path == "/mutate":
            current_ui_version = "redesigned"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "UI Mutated to v2.0"}).encode("utf-8"))
        elif self.path == "/reset":
            current_ui_version = "original"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "UI Reset to Original"}).encode("utf-8"))
        else:
            self.send_error(404)

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9000), MockBankHandler)
    print("Serving Mock Bank on http://localhost:9000 ...")
    server.serve_forever()