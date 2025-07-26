from flask import Flask, request, Response
import requests
import os

app = Flask(__name__)

# Configuration
FLASK_API_URL = "http://localhost:5000"
REACT_DEV_SERVER_URL = "http://localhost:3000"

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """Proxy API requests to Flask backend"""
    try:
        url = f"{FLASK_API_URL}/api/{path}"

        if request.method == 'GET':
            resp = requests.get(url, params=request.args)
        elif request.method == 'POST':
            if request.files:
                files = {}
                for key, file in request.files.items():
                    files[key] = (file.filename, file.stream, file.content_type)
                resp = requests.post(url, files=files, data=request.form)
            else:
                resp = requests.post(url, json=request.get_json(), params=request.args)
        elif request.method == 'PUT':
            resp = requests.put(url, json=request.get_json(), params=request.args)
        elif request.method == 'DELETE':
            resp = requests.delete(url, params=request.args)

        response = Response(
            resp.content,
            status=resp.status_code,
            headers=dict(resp.headers)
        )
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response
        
    except requests.exceptions.ConnectionError:
        return {"error": "Flask backend server is not running"}, 503
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy_react(path):
    """Proxy all other requests to React development server"""
    try:
        url = f"{REACT_DEV_SERVER_URL}/{path}"
        
        if request.method == 'GET':
            resp = requests.get(url, params=request.args, stream=True)
        else:
            return "Method not allowed", 405
        
        # Create response
        response = Response(
            resp.iter_content(chunk_size=1024),
            status=resp.status_code,
            headers=dict(resp.headers)
        )
        
        return response
        
    except requests.exceptions.ConnectionError:
        return "React development server is not running. Please start it with 'npm start'", 503
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Reverse proxy is running"}

if __name__ == '__main__':
    print("Starting reverse proxy server...")
    print(f"Proxying API requests to: {FLASK_API_URL}")
    print(f"Proxying frontend requests to: {REACT_DEV_SERVER_URL}")
    print("Server will be available at: http://localhost:8080")
    
    app.run(debug=True, host='0.0.0.0', port=8080)