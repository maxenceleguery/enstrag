from .agent_client import AgentClient
from .gradio_front import GradioFront

from flask import Flask, redirect, url_for, session
from flask_cas import CAS
import gradio as gr

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this!

# CAS Configuration
app.config['CAS_SERVER'] = 'https://cascad.ensta.fr'
app.config['CAS_AFTER_LOGIN'] = 'gradio_app'
app.config['CAS_LOGIN_ROUTE'] = '/login'
app.config['CAS_VALIDATE_ROUTE'] = '/p3/serviceValidate'
app.config['CAS_LOGOUT_ROUTE'] = '/logout'
app.config['CAS_SERVICE'] = 'http://127.0.0.1:5000/login'  # Adjust if running on another domain

cas = CAS(app)

# Protect Gradio App with CAS
@app.route('/')
def index():
    if 'CAS_USERNAME' not in session:
        return redirect(url_for('cas.login'))
    return redirect(url_for('gradio_app'))

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(app.config['CAS_SERVER'] + app.config['CAS_LOGOUT_ROUTE'])

# Serve Gradio App
@app.route('/gradio')
def gradio_app():
    if 'CAS_USERNAME' not in session:
        return redirect(url_for('cas.login'))
    
    agent = AgentClient()
    front = GradioFront(agent)

    # Launch Gradio inside Flask
    return front.launch(share=False, server_name="0.0.0.0", server_port=7861, inbrowser=False, prevent_thread_lock=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
