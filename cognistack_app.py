import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import io
import numpy as np
from flask import Flask, jsonify, request, render_template_string

# ==============================================================================
# 🧠 PART 1: THE MACHINE LEARNING MODEL (Cogni)
# ==============================================================================
# In a real app, this would be a separate, intensive training process.
# Here, we simulate it with dummy data for demonstration.

def create_and_train_models():
    """
    Creates dummy data and trains a separate Random Forest model for each
    of the 5 OCEAN personality traits. The trained models are returned.
    """
    print("🤖 Training machine learning models...")
    models = {}
    ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    # Create a dummy dataset
    # In reality, this would be thousands of survey results.
    num_samples = 100
    num_questions = 20
    X = np.random.randint(1, 6, size=(num_samples, num_questions))
    
    for trait in ocean_traits:
        # Create dummy labels (High/Low) for each trait
        # The logic here is random, just for demonstration
        y = np.random.choice(['High', 'Low'], size=num_samples, p=[0.5, 0.5])
        
        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save the model to an in-memory binary stream instead of a file
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        models[trait] = model_buffer
        print(f"✅ Model for '{trait}' trained successfully.")
        
    return models

# Train models on script startup
trained_models_in_memory = create_and_train_models()

# Function to load models from memory for prediction
def load_model_from_memory(trait):
    buffer = trained_models_in_memory[trait]
    buffer.seek(0)
    return joblib.load(buffer)

# ==============================================================================
# 💻 PART 2: THE BACKEND SERVER (Stack)
# ==============================================================================
# Using Flask to create a web server and API endpoints.

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to receive candidate answers and return personality analysis."""
    try:
        data = request.json
        answers = data.get('answers')
        
        # Basic validation
        if not answers or not isinstance(answers, list):
            return jsonify({"error": "Invalid answers format"}), 400

        # Ensure answers are in the correct format for the model (2D array)
        answers_vector = [answers]
        
        results = {}
        for trait in trained_models_in_memory.keys():
            model = load_model_from_memory(trait)
            prediction = model.predict(answers_vector)[0]
            probability = model.predict_proba(answers_vector)
            
            # Get the probability of the predicted class
            class_index = list(model.classes_).index(prediction)
            confidence = round(probability[0][class_index] * 100)
            
            results[trait] = {"level": prediction, "score": confidence}
            
        print(f"Analysis complete. Results: {results}")
        return jsonify(results)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": "An error occurred during analysis."}), 500

# ==============================================================================
# 🎨 PART 3: THE FRONTEND APPLICATION (UI/UX)
# ==============================================================================
# Serving an HTML file that contains the entire React.js application.
# React and Babel are loaded from a CDN for simplicity.

@app.route('/')
def home():
    """Serves the main HTML page containing the React app."""
    
    # This HTML string is the entire frontend.
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CogniStack Personality Assessment</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        body { 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh;
            margin: 0;
            color: #333;
        }
        #root { width: 100%; max-width: 600px; }
        .app-container { 
            background: #ffffff; 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        .question-card, .welcome-card, .results-card, .loading-card {
            animation: fadeIn 0.7s ease-in-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        h1 { color: #2c3e50; font-weight: 700; margin-bottom: 10px; }
        p { color: #555; line-height: 1.6; }
        .start-button, .scale-button {
            background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 51%, #6a11cb 100%);
            margin: 10px; padding: 15px 30px; text-align: center;
            text-transform: uppercase; transition: 0.5s; background-size: 200% auto;
            color: white; border-radius: 10px; display: inline-block;
            border: none; cursor: pointer; font-weight: 600;
        }
        .start-button:hover, .scale-button:hover { background-position: right center; }
        
        .progress-bar { width: 100%; background: #e0e0e0; border-radius: 5px; overflow: hidden; margin: 20px 0; }
        .progress-bar-inner { height: 10px; background-color: #2575fc; transition: width 0.4s ease-out; }

        .scale-container { display: flex; justify-content: space-between; align-items: center; margin-top: 30px; }
        .scale-button { padding: 12px 20px; }
        .scale-labels { display: flex; justify-content: space-between; padding: 0 10px; color: #777; font-size: 0.9em; }
        
        .results-grid { display: grid; grid-template-columns: 1fr; gap: 20px; text-align: left; margin-top: 30px; }
        .result-item { background: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #2575fc; }
        .result-item h3 { margin: 0 0 5px 0; color: #2c3e50; }
        .result-score { font-size: 1.2em; font-weight: 600; }
        .result-level-High { color: #27ae60; }
        .result-level-Low { color: #e74c3c; }

        .loader { border: 8px solid #f3f3f3; border-radius: 50%; border-top: 8px solid #2575fc; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect } = React;

        const questions = [
            "I enjoy exploring new ideas and theories.",
            "I am very organized and detail-oriented.",
            "I am the life of the party.",
            "I am sympathetic towards others' feelings.",
            "I rarely feel anxious or stressed.",
            "I prefer variety over routine.",
            "I am diligent and always complete my tasks.",
            "I feel comfortable around people.",
            "I tend to trust others easily.",
            "I can handle pressure well.",
            "I have a vivid imagination.",
            "I like to keep my things tidy.",
            "I don't mind being the center of attention.",
            "I enjoy helping others.",
            "I am emotionally stable.",
            "I am curious about many different things.",
            "I follow a schedule.",
            "I start conversations with strangers.",
            "I feel others' emotions.",
            "I am calm and relaxed most of the time."
        ];

        function App() {
            const [gameState, setGameState] = useState('welcome'); // welcome, assessment, loading, results
            const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
            const [answers, setAnswers] = useState([]);
            const [results, setResults] = useState(null);
            
            const handleStart = () => {
                setGameState('assessment');
            };

            const handleAnswer = (answer) => {
                const newAnswers = [...answers, answer];
                setAnswers(newAnswers);

                if (currentQuestionIndex < questions.length - 1) {
                    setCurrentQuestionIndex(currentQuestionIndex + 1);
                } else {
                    submitForAnalysis(newAnswers);
                }
            };

            const submitForAnalysis = async (finalAnswers) => {
                setGameState('loading');
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ answers: finalAnswers })
                    });
                    if (!response.ok) throw new Error('Network response was not ok');
                    const data = await response.json();
                    setResults(data);
                    setGameState('results');
                } catch (error) {
                    console.error("Error submitting analysis:", error);
                    alert("An error occurred. Please try again.");
                    setGameState('welcome'); // Reset on error
                }
            };
            
            const progress = (currentQuestionIndex / questions.length) * 100;

            return (
                <div className="app-container">
                    {gameState === 'welcome' && (
                        <div className="welcome-card">
                            <h1>Welcome to CogniStack</h1>
                            <p>This is a short assessment to provide insights into your work style preferences. Please answer honestly based on how you typically are.</p>
                            <p>There are {questions.length} questions in total.</p>
                            <button className="start-button" onClick={handleStart}>Start Assessment</button>
                        </div>
                    )}

                    {gameState === 'assessment' && (
                        <div className="question-card">
                            <h1>Question {currentQuestionIndex + 1}/{questions.length}</h1>
                            <div className="progress-bar">
                                <div className="progress-bar-inner" style={{ width: `${progress}%` }}></div>
                            </div>
                            <p style={{fontSize: '1.2em', minHeight: '60px'}}>{questions[currentQuestionIndex]}</p>
                            <div className="scale-labels">
                                <span>Strongly Disagree</span>
                                <span>Strongly Agree</span>
                            </div>
                            <div className="scale-container">
                                {[1, 2, 3, 4, 5].map(value => (
                                    <button className="scale-button" key={value} onClick={() => handleAnswer(value)}>
                                        {value}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                    
                    {gameState === 'loading' && (
                        <div className="loading-card">
                            <h1>Analyzing...</h1>
                            <p>Your results are being processed by our AI model.</p>
                            <div className="loader"></div>
                        </div>
                    )}

                    {gameState === 'results' && results && (
                        <div className="results-card">
                            <h1>Assessment Complete</h1>
                            <p>Here are your personality trait insights. This helps us understand how you might fit and thrive in our work environment.</p>
                            <div className="results-grid">
                                {Object.entries(results).map(([trait, data]) => (
                                    <div className="result-item" key={trait}>
                                        <h3>{trait}</h3>
                                        <p>
                                            <span className={`result-score result-level-${data.level}`}>{data.level}</span>
                                            {' '} (Score: {data.score}%)
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
    """
    return render_template_string(html_template)


# ==============================================================================
# 🚀 PART 4: RUN THE APPLICATION
# ==============================================================================

if __name__ == '__main__':
    print("=========================================================")
    print("🚀 CogniStack Server is starting...")
    print("🌍 Access the application at: http://127.0.0.1:5000")
    print("=========================================================")
    # Using host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False)