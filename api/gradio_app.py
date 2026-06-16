import gradio as gr
import joblib
import numpy as np

model = joblib.load('models/saved/final_model.pkl')
threshold = joblib.load('models/saved/best_threshold.pkl')

def predict(num_views, added_to_cart, session_duration_min, 
            total_items_viewed, hour_of_day, day_of_week):
    added_to_cart = 1 if added_to_cart == "Yes" else 0
    
    features = np.array([[num_views, added_to_cart, session_duration_min,
                          total_items_viewed, hour_of_day, day_of_week]])
    
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= threshold)
    
    result = "✅ Will Purchase" if prediction == 1 else "❌ Will NOT Purchase"
    return f"{result}\nProbability: {prob:.2%}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Number of Views"),
        gr.Radio(["No", "Yes"], label="Added to Cart"),
        gr.Number(label="Session Duration (minutes)"),
        gr.Number(label="Total Items Viewed"),
        gr.Slider(0, 23, step=1, label="Hour of Day"),
        gr.Slider(0, 6, step=1, label="Day of Week (0=Mon, 6=Sun)")
    ],
    outputs=gr.Text(label="Prediction"),
    title="E-commerce Conversion Prediction",
    description="Predict whether a visitor will make a purchase"
)

demo.launch()