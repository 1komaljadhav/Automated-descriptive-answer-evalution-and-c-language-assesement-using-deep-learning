from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import subprocess
import logging
import math

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@app.template_filter('ceil')
def ceil_filter(value):
    return math.ceil(value)
# Load the trained model
try:
    model = pickle.load(open('answer_evaluation_model.pkl', 'rb'))
except FileNotFoundError:
    raise Exception("Model file not found.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Load the trained tokenizer
try:
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
except FileNotFoundError:
    raise Exception("Tokenizer file not found.")
except Exception as e:
    raise Exception(f"Error loading tokenizer: {str(e)}")

# Load dataset
try:
    df = pd.read_csv('student_evaluation_results (2).csv', encoding='ISO-8859-1')
except FileNotFoundError:
    raise Exception("Dataset file not found.")
except Exception as e:
    raise Exception(f"Error loading dataset: {str(e)}")

max_seq_length = 100

# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to compute keyword overlap
def keyword_overlap(answer, model_answer):
    clean_answer = remove_stopwords(answer)
    clean_model_answer = remove_stopwords(model_answer)

    answer_keywords = set(clean_answer.lower().split())
    model_answer_keywords = set(clean_model_answer.lower().split())

    overlap = answer_keywords.intersection(model_answer_keywords)
    return len(overlap)

@app.route('/')
def index():
    try:
        random_questions = df['Questions'].sample(n=5).tolist()
    except Exception as e:
        return f"Error fetching questions: {str(e)}", 500

    return render_template('index.html', questions=random_questions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        answers = {}
        total_marks = 0
        max_marks_per_question = 10
        total_max_marks = 50

        questions = []
        student_answers = []

        for i in range(1, 6):
            question_key = f'que{i}'
            answer_key = f'ans{i}'

            if question_key in request.form and answer_key in request.form:
                question = request.form[question_key]
                answer = request.form[answer_key]

                logging.debug(f"Question {i}: {question}")
                logging.debug(f"Answer {i}: {answer}")

                questions.append(question)
                student_answers.append(answer)

        question_seqs = tokenizer.texts_to_sequences(questions)
        answer_seqs = tokenizer.texts_to_sequences(student_answers)

        question_seqs = pad_sequences(question_seqs, maxlen=max_seq_length)
        answer_seqs = pad_sequences(answer_seqs, maxlen=max_seq_length)

        logging.debug(f"Question Sequences Shape: {question_seqs.shape}")
        logging.debug(f"Answer Sequences Shape: {answer_seqs.shape}")

        predicted_marks = model.predict([question_seqs, answer_seqs])
        logging.debug(f"Predicted Marks: {predicted_marks}")

        vectorizer = CountVectorizer(stop_words='english')
        model_answers = df['Answers'].tolist()
        vectorizer.fit(model_answers)

        for i, question in enumerate(questions):
            answer = student_answers[i].strip()

            if not answer or len(answer.split()) < 5:
                marks = 0
            else:
                marks = min(max(0, round(predicted_marks[i][0], 2)), max_marks_per_question)
                
                question_model_answers = df[df['Questions'].str.lower() == question.lower()]['Answers'].tolist()
                overlaps = [keyword_overlap(answer, model_answer) for model_answer in question_model_answers]
                
                max_overlap = max(overlaps, default=0)
                if max_overlap < 4:
                    marks = 0

            answers[question] = {'answer': answer, 'marks': marks}
            total_marks += marks
        
        normalized_total_marks = min(total_marks, total_max_marks)
        
        return render_template('after.html', data=answers, total_marks=normalized_total_marks, max_marks=total_max_marks)
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/up')
def up():
    return render_template('upload.html')
SAVE_DIR = 'C:/workspaces/error_log/approch_1/test'

# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    c_code = request.form.get('c_code')
    input_file = request.files.get('input_file')

    # Paths for saving the files
    code_file_path = 'C:/workspaces/error_log/approch_1/test/code.c'
    input_file_path = 'C:/workspaces/error_log/approch_1/test/input.txt'

    # Save the C code to a file
    if c_code:
        with open(code_file_path, 'w', encoding='utf-8') as code_file:
            code_file.write(c_code)

    # Save the uploaded input file
    if input_file:
        input_file.save(input_file_path)

    # Compile and run the C program
    status, message = run_c_program(code_file_path)

    if status in ['Compilation Error', 'Runtime Error']:
        marks = calculate_marks(message)
        return render_template('result.html', status=status, message=message, marks=marks)
    else:
        return render_template('result.html', status=status, message=message, marks=10)

import subprocess
import os

def evaluate_errors(error_log):
    """
    Evaluates the error log and classifies errors into severity and type.
    
    Parameters:
    - error_log (str): The error log output from the compilation or runtime process.
    
    Returns:
    - dict: A dictionary with error details.
    """
    errors = {
        "total_errors": 0,
        "critical_errors": 0,
        "warnings": 0,
        "error_messages": []
    }

    # Example criteria: Counting critical errors and warnings
    for line in error_log.splitlines():
        if "error:" in line.lower():
            errors["total_errors"] += 1
            errors["critical_errors"] += 1
            errors["error_messages"].append(line)
        elif "warning:" in line.lower():
            errors["total_errors"] += 1
            errors["warnings"] += 1
            errors["error_messages"].append(line)
    
    return errors
import os
import subprocess

def run_c_program(file_path):
    try:
        # Save output to a directory with confirmed write permissions
        output_file_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'a.exe')
        
        gcc_path = 'C:/Users/DELL/Downloads/gcc-14.1.0-no-debug/bin/gcc-14.1.0'

        # Compile the C program
        compile_process = subprocess.Popen(
            [gcc_path, '-o', output_file_path, file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = compile_process.communicate()

        if compile_process.returncode != 0:
            error_log = stderr.decode('utf-8')
            return "Compilation Error", classify_errors(error_log, is_runtime=False)
        
        # Try to run the compiled program to check for runtime errors
        run_process = subprocess.Popen(
            [output_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        run_stdout, run_stderr = run_process.communicate()

        if run_process.returncode != 0:
            runtime_error_log = run_stderr.decode('utf-8')
            return "Runtime Error", classify_errors(runtime_error_log, is_runtime=True)
        
        # Return program output
        output = run_stdout.decode('utf-8')
        return "Success", output

    except Exception as e:
        return "Runtime Error", str(e)

def classify_errors(error_log, is_runtime=False):
    error_details = {
        'total_errors': 0,
        'critical_errors': 0,
        'syntax_errors': 0,
        'warnings': 0,
        'error_messages': []
    }

    lines = error_log.split('\n')
    for line in lines:
        if 'error' in line.lower():
            error_details['total_errors'] += 1
            if 'critical' in line.lower():
                error_details['critical_errors'] += 1
            elif 'syntax' in line.lower():
                error_details['syntax_errors'] += 1
            else:
                error_details['warnings'] += 1
            error_details['error_messages'].append(line)
    
    if is_runtime:
        error_details['warnings'] += len(error_details['error_messages'])

    return error_details

def calculate_marks(error_details):
    max_marks = 10
    total_errors = error_details['total_errors']
    critical_errors = error_details['critical_errors']
    syntax_errors = error_details['syntax_errors']
    warnings = error_details['warnings']

    # Define the impact of each type of error
    critical_error_penalty = max_marks
    syntax_error_penalty = max_marks * 0.5
    warning_penalty = max_marks * 0.2

    # Calculate marks based on errors
    if critical_errors > 0:
        return 0  # All critical errors result in 0 marks

    marks = max_marks - (syntax_errors * syntax_error_penalty) - (warnings * warning_penalty)
    marks = max(0, marks)  # Ensure marks are not negative

    return round(marks, 2)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
