from flask import Flask, request, render_template

app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling the form submission
@app.route('/submit', methods=['POST'])
def submit():
    question = request.form['question']
    # Call your backend API to get the result
    result = get_prediction(question)
    return render_template('result.html', question=question, result=result)

def get_prediction(question):
    return "result from backend"

if __name__ == '__main__':
    app.run(debug=True)