from flask import Flask, redirect, url_for, session, render_template
from flask_cas import CAS
from functools import wraps

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this!

# CAS Configuration
app.config['CAS_SERVER'] = 'https://cascad.ensta.fr'
app.config['CAS_AFTER_LOGIN'] = '/gradio'
app.config['CAS_LOGIN_ROUTE'] = '/login'
app.config['CAS_VALIDATE_ROUTE'] = '/p3/serviceValidate'
app.config['CAS_LOGOUT_ROUTE'] = '/logout'
#app.config['CAS_SERVICE'] = 'http://127.0.0.1:5000/login'  # Adjust if running on another domain

cas = CAS(app)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'CAS_USERNAME' not in session:
            return redirect(url_for('cas.login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'CAS_USERNAME' in session:
        return redirect(url_for('auth'))

    return render_template('index.html', login_url='/auth')
    #return render_template('index.html', login_url=url_for('cas.login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(app.config['CAS_SERVER'] + app.config['CAS_LOGOUT_ROUTE'])

@app.route('/auth/')
@login_required
def auth():
    #if auth successful
    return redirect('/enstrag')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
