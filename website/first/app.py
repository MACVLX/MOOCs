from flask import Flask, render_template, request, url_for

# Configure application
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/project_1')
def project_1():
    return render_template('project_1.html')


if __name__=='__main__':
    app.run(debug=True)