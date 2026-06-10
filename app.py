"""
app.py — Main Flask Application
Mental Health AI Support System
"""
from dotenv import load_dotenv

load_dotenv()

import os
import pickle
import json
import threading
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

# Sentiment Analysis and AI Clients
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai import errors


# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mh-ai-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# ── Load ML Model ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join('models', 'risk_model.pkl')
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# ── Database Models ──────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    responses    = db.relationship('QuestionnaireResponse', backref='user', lazy=True)
    appointments = db.relationship('Appointment', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class QuestionnaireResponse(db.Model):
    __tablename__ = 'questionnaire_responses'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    sleep_hours     = db.Column(db.Float, nullable=False)
    stress_level    = db.Column(db.Integer, nullable=False)
    study_pressure  = db.Column(db.Integer, nullable=False)
    social_anxiety  = db.Column(db.Integer, nullable=False)
    mood_score      = db.Column(db.Integer, nullable=False)
    feelings_text   = db.Column(db.Text)
    risk_level      = db.Column(db.String(20))   # Low / Medium / High / Critical
    risk_score      = db.Column(db.Integer)       # 0, 1, 2, 3
    recommendation  = db.Column(db.Text)
    submitted_at    = db.Column(db.DateTime, default=datetime.utcnow)

    # Expanded Resume-worthy Features
    sentiment_score     = db.Column(db.Float, default=0.0)
    sentiment_label     = db.Column(db.String(50), default="Neutral")
    crisis_detected     = db.Column(db.Boolean, default=False)
    sleep_goal          = db.Column(db.Float)
    walk_goal           = db.Column(db.Integer)
    meditation_goal     = db.Column(db.Integer)
    journaling_goal     = db.Column(db.Integer)
    explainable_ai_json = db.Column(db.Text)  # JSON string
    rec_resources_json  = db.Column(db.Text)  # JSON string



class Doctor(db.Model):
    __tablename__ = 'doctors'
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    specialization= db.Column(db.String(100))
    location      = db.Column(db.String(200))
    available_days= db.Column(db.String(200))
    phone         = db.Column(db.String(50))
    email         = db.Column(db.String(120))
    about         = db.Column(db.Text)
    rating        = db.Column(db.Float, default=4.0)
    tags          = db.Column(db.String(200))  # e.g. "anxiety,depression,stress"
    experience_years = db.Column(db.Integer, default=5)
    appointments  = db.relationship('Appointment', backref='doctor', lazy=True)


class Appointment(db.Model):
    __tablename__ = 'appointments'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor_id     = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    date          = db.Column(db.String(20), nullable=False)
    time_slot     = db.Column(db.String(20), nullable=False)
    reason        = db.Column(db.Text)
    status        = db.Column(db.String(20), default='Pending')  # Pending/Confirmed/Cancelled
    booked_at     = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Helper ───────────────────────────────────────────────────────────────────

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
RISK_COLORS = {0: 'success', 1: 'warning', 2: 'danger', 3: 'critical'}

RECOMMENDATIONS = {
    0: ("Great job taking care of yourself! Keep up the healthy routine. "
        "Maintain regular sleep, exercise, and social connections."),
    1: ("Your responses indicate moderate stress. Consider trying mindfulness, "
        "talking to a friend, or scheduling a counseling session to help manage your wellbeing."),
    2: ("Your responses indicate high stress and risk. We strongly recommend speaking "
        "with a counselor or mental health professional as soon as possible. You are not alone — help is available."),
    3: ("🚨 Critical risk level detected. We strongly advise reaching out to emergency support services or a "
        "crisis helpline immediately. Immediate professional care is highly recommended. You are not alone.")
}

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_and_crisis(text):
    """
    Analyzes input text to determine compound sentiment score, label, and crisis indicators.
    """
    if not text or not text.strip():
        return 0.0, "Neutral / Reflective", False

    # 1. Crisis Detection
    crisis_keywords = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die", 
        "disappear", "don't want to live", "self harm", "harm myself", 
        "want to end it", "disappearing forever", "no point in living"
    ]
    text_lower = text.lower()
    crisis_detected = any(kw in text_lower for kw in crisis_keywords)

    # 2. Sentiment analysis
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # Map compound sentiment score to emotional state labels
    if compound <= -0.5:
        label = "Distressed / Highly Concerned"
    elif compound <= -0.05:
        label = "Anxious / Overwhelmed"
    elif compound < 0.05:
        label = "Neutral / Reflective"
    elif compound < 0.5:
        label = "Positive / Stable"
    else:
        label = "Excellent / Calm"

    return compound, label, crisis_detected


def calculate_hybrid_risk_and_explain(sleep, stress, pressure, anxiety, mood, feelings_text):
    """
    Combines questionnaire metrics, scikit-learn model probability, and sentiment analysis 
    to calculate a unified Hybrid Risk Score, Explainable AI deviations, and recovery goals.
    """
    # 1. Run Sentiment & Crisis Detection
    compound, sentiment_label, crisis_detected = analyze_sentiment_and_crisis(feelings_text)

    # 2. Base RF Score from Random Forest Model if available, else fallback rule
    if model is not None:
        import pandas as pd
        features = pd.DataFrame(
            [[float(sleep), float(stress), float(pressure), float(anxiety), float(mood)]],
            columns=['sleep_hours', 'stress_level', 'study_pressure', 'social_anxiety', 'mood_score']
        )
        probs = model.predict_proba(features)[0]  # probabilities for [Low, Medium, High]
        # Calculate continuous base RF score (0 to 100)
        rf_score = float((probs[1] * 50.0) + (probs[2] * 100.0))
        pred_class = int(model.predict(features)[0])
    else:
        # Fallback simple rule base score
        avg_symptoms = (stress + pressure + anxiety) / 3.0
        if sleep < 5 or avg_symptoms >= 7:
            pred_class = 2
        elif sleep < 7 or avg_symptoms >= 5:
            pred_class = 1
        else:
            pred_class = 0
        rf_score = pred_class * 50.0

    # 3. Penalties calculation for hybrid score (Upgrade 10)
    # Sleep penalty: deviance from healthy sleep (7.5 hours)
    sleep_penalty = 0.0
    if sleep < 7.5:
        sleep_penalty = min(100.0, ((7.5 - sleep) / 4.5) * 100.0)
    elif sleep > 9.0:
        sleep_penalty = min(30.0, ((sleep - 9.0) / 3.0) * 100.0)

    # Stress penalty: stress level is 1 to 10. Map to 0-100.
    stress_penalty = float(stress * 10.0)

    # Sentiment penalty: negative sentiment increases risk penalty
    sentiment_penalty = 0.0
    if compound < 0:
        sentiment_penalty = float(abs(compound) * 100.0)

    # 4. Combine into Hybrid Score (0 to 100)
    # Weights: RF (40%), Stress (30%), Sentiment (20%), Sleep (10%)
    hybrid_index = (0.4 * rf_score) + (0.3 * stress_penalty) + (0.2 * sentiment_penalty) + (0.1 * sleep_penalty)

    # Crisis override
    if crisis_detected:
        hybrid_index = 95.0

    # Map to final levels: Low (0), Medium (1), High (2), Critical (3)
    if hybrid_index < 35:
        final_score = 0
        final_level = "Low"
    elif hybrid_index < 65:
        final_score = 1
        final_level = "Medium"
    elif hybrid_index < 85:
        final_score = 2
        final_level = "High"
    else:
        final_score = 3
        final_level = "Critical"

    # 5. Explainable AI calculations (contribution percentages)
    # Calculate relative deviance of each factor from healthy baseline
    dev_sleep = max(0.0, 7.5 - sleep) / 7.5 * 100.0
    dev_stress = max(0.0, stress - 3.0) / 7.0 * 100.0
    dev_pressure = max(0.0, pressure - 3.0) / 7.0 * 100.0
    dev_anxiety = max(0.0, anxiety - 3.0) / 7.0 * 100.0
    dev_mood = max(0.0, 7.0 - mood) / 7.0 * 100.0
    dev_sentiment = float(max(0.0, -compound) * 100.0)

    # Total deviance for normalization
    total_dev = dev_sleep + dev_stress + dev_pressure + dev_anxiety + dev_mood + dev_sentiment
    if total_dev > 0:
        contrib_sleep = round((dev_sleep / total_dev) * 100.0)
        contrib_stress = round((dev_stress / total_dev) * 100.0)
        contrib_pressure = round((dev_pressure / total_dev) * 100.0)
        contrib_anxiety = round((dev_anxiety / total_dev) * 100.0)
        contrib_mood = round((dev_mood / total_dev) * 100.0)
        contrib_sentiment = round((dev_sentiment / total_dev) * 100.0)
    else:
        contrib_sleep = contrib_stress = contrib_pressure = contrib_anxiety = contrib_mood = contrib_sentiment = 0

    explain_dict = {
        "Poor Sleep": contrib_sleep,
        "High Stress": contrib_stress,
        "Study Pressure": contrib_pressure,
        "Social Anxiety": contrib_anxiety,
        "Low Mood": contrib_mood,
        "Negative Feelings": contrib_sentiment
    }

    # Keep only positive contributors for clear display
    explain_dict = {k: v for k, v in explain_dict.items() if v > 0}

    # 6. Personalized Recovery Plan goals (Upgrade 1)
    sleep_goal = 8.0 if sleep < 7.0 else 7.5
    if final_score == 0:
        walk_goal = 15
        meditation_goal = 5
        journaling_goal = 5
    elif final_score == 1:
        walk_goal = 20
        meditation_goal = 10
        journaling_goal = 5
    else:
        walk_goal = 25
        meditation_goal = 10
        journaling_goal = 10

    # 7. Resource Recommendations (Upgrade 5)
    resources = []
    feelings_lower = feelings_text.lower()
    
    if sleep < 7.0 or "sleep" in feelings_lower or "tired" in feelings_lower or "insomnia" in feelings_lower:
        resources.append({
            "title": "Sleep Hygiene Guidelines",
            "desc": "Establish a calming evening routine. Avoid screens for at least 1 hour before bed.",
            "link": "https://www.sleepfoundation.org/sleep-hygiene"
        })
        resources.append({
            "title": "Guided Sleep Meditation Video",
            "desc": "Listen to a 10-minute relaxing body scan to ease into deep sleep.",
            "link": "https://www.youtube.com/watch?v=v7AYKzSoHBM"
        })
        
    if stress > 5 or pressure > 5 or "exam" in feelings_lower or "study" in feelings_lower or "pressure" in feelings_lower:
        resources.append({
            "title": "Time Management Tool: Pomodoro Method",
            "desc": "Study for 25 minutes, then take a 5-minute break. This prevents mental fatigue.",
            "link": "https://todoist.com/productivity-methods/pomodoro-technique"
        })
        resources.append({
            "title": "Managing Academic Stress Guide",
            "desc": "Tips on planning, chunking study materials, and managing performance anxiety.",
            "link": "https://www.jedfoundation.org/resource/managing-academic-stress/"
        })
        
    if anxiety > 5 or "anxious" in feelings_lower or "nervous" in feelings_lower or "panic" in feelings_lower:
        resources.append({
            "title": "5-4-3-2-1 Grounding Method",
            "desc": "Identify 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste.",
            "link": "https://www.urmc.rochester.edu/behavioral-health-partners/register/billing-insurance/grounding-techniques.aspx"
        })
        resources.append({
            "title": "Box Breathing Calm Exercise",
            "desc": "Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s to regulate your heart rate.",
            "link": "https://www.healthline.com/health/box-breathing"
        })
        
    if mood < 5 or "hopeless" in feelings_lower or "sad" in feelings_lower or "lonely" in feelings_lower:
        resources.append({
            "title": "Behavioral Activation Guide",
            "desc": "Schedule one small pleasant activity today (e.g. listening to music or texting a friend).",
            "link": "https://www.healthline.com/health/depression/behavioral-activation"
        })
        resources.append({
            "title": "Daily Gratitude Journaling Tips",
            "desc": "Write down 3 tiny things that went well today. Focus on sensory and small pleasures.",
            "link": "https://greatergood.berkeley.edu/article/item/how_to_cultivate_gratitude_in_difficult_times"
        })

    # Default general resources if none matched
    if not resources:
        resources.append({
            "title": "Mindfulness & Breathing Exercises",
            "desc": "Simple exercises to bring your focus back to the present moment.",
            "link": "https://www.mindful.org/meditation/mindfulness-getting-started/"
        })
        resources.append({
            "title": "Healthy Self-Care Routine Guide",
            "desc": "How nutrition, light exercise, and social connection build strong mental resilience.",
            "link": "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health"
        })

    return {
        "hybrid_index": hybrid_index,
        "final_score": final_score,
        "final_level": final_level,
        "sentiment_score": compound,
        "sentiment_label": sentiment_label,
        "crisis_detected": crisis_detected,
        "sleep_goal": sleep_goal,
        "walk_goal": walk_goal,
        "meditation_goal": meditation_goal,
        "journaling_goal": journaling_goal,
        "explainable_ai": explain_dict,
        "rec_resources": resources
    }


def recommend_doctors(risk_score, user_location=None):
    """
    Returns the top 3 doctors/clinics matched to the user's risk level.
    Upgrade 4: Low -> Self-help (shown in view), Med -> Therapist, High -> Psychiatrist, Critical -> Crisis Care
    """
    doctors = Doctor.query.all()
    scored = []

    for doc in doctors:
        score = 0
        tags = (doc.tags or "").lower()
        spec = (doc.specialization or "").lower()

        # Match profiles based on risk score
        if risk_score == 3:  # Critical
            # Prefer senior psychiatric specialists and hospitals
            if "psychiatrist" in spec or "senior" in spec or "hospital" in spec:
                score += 10
            if doc.experience_years >= 15:
                score += 5
        elif risk_score == 2:  # High
            if "psychiatrist" in spec:
                score += 8
            if "depression" in tags or "anxiety" in tags:
                score += 5
        elif risk_score == 1:  # Medium
            # Prefer therapist/counseling
            if "therapy" in tags or "ocd" in tags or "stress" in tags:
                score += 8
            if "counselor" in spec or "therapist" in spec:
                score += 5
        else:  # Low
            score += 2

        # Add rating weight
        score += (doc.rating or 4.0) * 2.0
        # Experience weight
        score += (doc.experience_years or 5) * 0.3

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:3]]


def send_email_async(to_email, user_name, result_data, doctor_name, doctor_contact):
    """
    Sends an assessment report email in a background thread.
    If credentials are default/empty, logs report content to instance/email_logs.txt.
    """
    subject = f"Your MindCare AI Wellness Report - {result_data['final_level']} Risk"

    # HTML elements construction
    plan_html = f"""
    <ul>
        <li>😴 <strong>Sleep Goal:</strong> {result_data['sleep_goal']} hours</li>
        <li>🚶 <strong>Daily Walk:</strong> {result_data['walk_goal']} minutes</li>
        <li>🧘 <strong>Meditation:</strong> {result_data['meditation_goal']} minutes</li>
        <li>✍️ <strong>Journaling:</strong> {result_data['journaling_goal']} minutes</li>
    </ul>
    """
    
    resources_html = "".join([
        f"<li><strong><a href='{res['link']}'>{res['title']}</a></strong>: {res['desc']}</li>"
        for res in result_data['rec_resources']
    ])

    doctor_html = ""
    if result_data['final_score'] > 0:
        doctor_html = f"""
        <h3>👨‍⚕️ Recommended Professional Support:</h3>
        <p>Based on your profile, we recommend reaching out to:</p>
        <p><strong>{doctor_name}</strong><br>Contact: {doctor_contact}</p>
        """

    crisis_html = ""
    if result_data['crisis_detected']:
        crisis_html = """
        <div style="background-color: #fee2e2; border: 2px solid #ef4444; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #991b1b;">
            <h3 style="margin-top: 0;">🚨 Immediate Crisis Support Available</h3>
            <p>If you are struggling or in distress, please connect with a helpline immediately:</p>
            <ul>
                <li><strong>Tele-MANAS (India):</strong> 14416 or 1800-891-4416 (24/7 Free helpline)</li>
                <li><strong>Kiran Mental Health Helpline:</strong> 1800-599-0019</li>
                <li><strong>Vandrevala Foundation:</strong> +91 9999 666 555</li>
            </ul>
            <p>You matter, and support is available whenever you need it.</p>
        </div>
        """

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #6366f1;">🧠 MindCare AI Assessment Report</h2>
        <p>Hello {user_name},</p>
        <p>Thank you for completing your wellness assessment today. Below is a copy of your report for your records.</p>
        
        {crisis_html}

        <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #6366f1;">
            <h3 style="margin-top: 0; color: #4b5563;">📊 Assessment Summary:</h3>
            <p style="margin: 5px 0;"><strong>Risk Level:</strong> {result_data['final_level']}</p>
            <p style="margin: 5px 0;"><strong>Date:</strong> {datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')}</p>
            <p style="margin: 5px 0;"><strong>Reported Sleep:</strong> {result_data['sleep_hours']} hours</p>
            <p style="margin: 5px 0;"><strong>Mood Rating:</strong> {result_data['mood_score']}/10</p>
            <p style="margin: 5px 0;"><strong>Open Feelings Sentiment:</strong> {result_data['sentiment_label']}</p>
        </div>

        <h3 style="color: #4f46e5;">🌱 Your Personalized Recovery Plan:</h3>
        {plan_html}

        <h3 style="color: #4f46e5;">📚 Recommended Resources:</h3>
        <ul>
            {resources_html}
        </ul>

        {doctor_html}

        <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0;">
        <p style="font-size: 0.85em; color: #6b7280; text-align: center;">
            This report was auto-generated by MindCare AI. If you are experiencing a medical or psychiatric emergency, please visit your local hospital emergency room or contact emergency services immediately.
        </p>
    </body>
    </html>
    """

    sender_email = os.environ.get("GMAIL_ADDRESS", "").strip()
    sender_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if not sender_email or not sender_password or "your_app_password" in sender_password or not to_email:
        # Fallback to local logging
        os.makedirs("instance", exist_ok=True)
        log_file = os.path.join("instance", "email_logs.txt")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n==================================================\n")
                f.write(f"EMAIL TO: {to_email}\n")
                f.write(f"SUBJECT: {subject}\n")
                f.write(f"DATE: {datetime.utcnow().isoformat()}\n")
                f.write(f"==================================================\n")
                f.write(html_content)
                f.write(f"\n==================================================\n")
            print(f"[Mock Email Service] Credentials not configured. Report written to {log_file}")
        except Exception as e:
            print(f"[ERROR] Error logging email to file: {e}")
        return

    # Real SMTP thread execution
    def send_process():
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = to_email

            part_html = MIMEText(html_content, "html")
            msg.attach(part_html)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, to_email, msg.as_string())
            print(f"[SUCCESS] Email successfully sent to {to_email}")
        except Exception as e:
            print(f"[ERROR] Error sending email: {e}")

    threading.Thread(target=send_process, daemon=True).start()


def get_simulated_ai_response(user_input):
    """
    Conversational fallback state engine that acts like an empathetic wellness counselor.
    """
    text = user_input.lower()
    
    # 1. Stress & Overwhelm
    if any(kw in text for kw in ["stress", "stressed", "overwhelmed", "pressure", "exam", "deadline", "study"]):
        return ("I hear you, and it's completely normal to feel overwhelmed when facing academic pressure. "
                "Try utilizing the **Pomodoro Technique**: study for 25 minutes, then take a 5-minute break. "
                "Let's do a deep breath right now: inhale for 4 seconds, hold for 4, and exhale for 4. "
                "You are doing your best, and your grades do not define your worth as a human. 💙")
                
    # 2. Depression & Hopelessness
    if any(kw in text for kw in ["sad", "depressed", "unhappy", "hopeless", "crying", "miserable"]):
        return ("I'm really sorry you are feeling this way. It takes a lot of courage to share these feelings. "
                "When mood is low, small steps are very powerful. Try scheduling just one tiny, pleasant activity "
                "today, like listening to a favorite track, looking outside, or texting a friend. "
                "Please consider booking an appointment with one of our counselors. You don't have to carry this alone. 💙")
                
    # 3. Sleep Issues
    if any(kw in text for kw in ["sleep", "insomnia", "tired", "exhausted", "awake", "nightmare"]):
        return ("Rest is so important for your mental wellbeing. Try establishing a consistent sleep routine "
                "and keeping screens away for 1 hour before bed. "
                "If your mind is racing, try writing down all your worries on a physical notepad "
                "to 'store' them away until tomorrow. Easing into a body scan meditation can also help. 😴")
                
    # 4. Anxiety & Panic
    if any(kw in text for kw in ["anxious", "anxiety", "nervous", "panic", "scared", "worried"]):
        return ("Anxiety can feel incredibly intense in the body. Let's do a quick **5-4-3-2-1 grounding exercise**: "
                "name 5 things you see around you, 4 things you can touch, 3 things you hear, "
                "2 things you smell, and 1 thing you taste. This helps bring your brain back to safety in the present. "
                "Remember, this feeling will pass. You are safe. 💙")
                
    # 5. Loneliness & Isolation
    if any(kw in text for kw in ["lonely", "alone", "isolated", "no friends", "nobody"]):
        return ("Loneliness can be very painful, but please remember you are not alone in this. "
                "Even a very brief interaction—like saying hi to a peer or checking in on a family member—can "
                "help break the cycle of isolation. Joining student groups or talking to a counselor can also help. "
                "I am always here to listen to you. 🤝")

    # 6. Self-care and wellness
    if any(kw in text for kw in ["self-care", "wellness", "routine", "exercise", "meditation"]):
        return ("Self-care is built on small daily habits: eating regular meals, drinking water, taking short walks, "
                "and practicing self-compassion. Try setting a timer for a 15-minute walk today or doing 5 minutes "
                "of mindfulness. Small wins compound into big health benefits!")

    # 7. Greetings
    if any(kw in text for kw in ["hi", "hello", "hey", "greetings"]):
        return "Hello! 👋 I'm MindBot, your mental health support assistant. How are you feeling today? Tell me what's on your mind."

    # 8. Help / queries
    if any(kw in text for kw in ["help", "support", "counselor", "doctor", "appointment"]):
        return ("If you need human support, you can book an appointment with our counselors under the **Appointments** page. "
                "If you are experiencing a crisis, please call Tele-MANAS (14416) or Kiran (1800-599-0019) immediately.")

    # Generic fallback
    return ("Thank you for sharing that with me. It sounds like you are navigating some complex thoughts. "
            "I'm here to listen. Could you tell me more about what is contributing to this feeling, "
            "or what has helped you feel supported in the past? 💙")



# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([name, email, password, confirm]):
            flash('All fields are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered. Please log in.', 'warning')
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    responses = QuestionnaireResponse.query.filter_by(
        user_id=current_user.id).order_by(
        QuestionnaireResponse.submitted_at.desc()).all()
    latest = responses[0] if responses else None
    appointments = Appointment.query.filter_by(
        user_id=current_user.id).order_by(
        Appointment.booked_at.desc()).limit(3).all()
    return render_template('dashboard.html',
                           responses=responses,
                           latest=latest,
                           appointments=appointments)


@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    if request.method == 'POST':
        try:
            sleep    = float(request.form['sleep_hours'])
            stress   = int(request.form['stress_level'])
            pressure = int(request.form['study_pressure'])
            anxiety  = int(request.form['social_anxiety'])
            mood     = int(request.form['mood_score'])
            feelings = request.form.get('feelings_text', '')

            # Calculate the hybrid model and extract all details
            res = calculate_hybrid_risk_and_explain(sleep, stress, pressure, anxiety, mood, feelings)

            resp = QuestionnaireResponse(
                user_id=current_user.id,
                sleep_hours=sleep,
                stress_level=stress,
                study_pressure=pressure,
                social_anxiety=anxiety,
                mood_score=mood,
                feelings_text=feelings,
                risk_level=res['final_level'],
                risk_score=res['final_score'],
                recommendation=RECOMMENDATIONS[res['final_score']],
                
                # New metrics
                sentiment_score=res['sentiment_score'],
                sentiment_label=res['sentiment_label'],
                crisis_detected=res['crisis_detected'],
                sleep_goal=res['sleep_goal'],
                walk_goal=res['walk_goal'],
                meditation_goal=res['meditation_goal'],
                journaling_goal=res['journaling_goal'],
                explainable_ai_json=json.dumps(res['explainable_ai']),
                rec_resources_json=json.dumps(res['rec_resources'])
            )
            db.session.add(resp)
            db.session.commit()

            # Recommend doctors/therapists
            recd_docs = recommend_doctors(res['final_score'])
            doc_name = recd_docs[0].name if recd_docs else "N/A"
            doc_contact = recd_docs[0].phone if recd_docs else "N/A"

            # Trigger Asynchronous Email Reports (Upgrade 8)
            send_email_async(
                to_email=current_user.email,
                user_name=current_user.name,
                result_data={
                    'final_level': res['final_level'],
                    'final_score': res['final_score'],
                    'sleep_hours': sleep,
                    'mood_score': mood,
                    'sentiment_label': res['sentiment_label'],
                    'sleep_goal': res['sleep_goal'],
                    'walk_goal': res['walk_goal'],
                    'meditation_goal': res['meditation_goal'],
                    'journaling_goal': res['journaling_goal'],
                    'rec_resources': res['rec_resources'],
                    'crisis_detected': res['crisis_detected']
                },
                doctor_name=doc_name,
                doctor_contact=doc_contact
            )

            return redirect(url_for('result', response_id=resp.id))
        except (KeyError, ValueError) as e:
            print(f"Form submission error: {e}")
            flash('Please fill in all fields correctly.', 'danger')
    return render_template('questionnaire.html')


@app.route('/result/<int:response_id>')
@login_required
def result(response_id):
    resp = QuestionnaireResponse.query.get_or_404(response_id)

    if resp.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))

    color = RISK_COLORS.get(resp.risk_score, 'info')
    recommended_doctors = recommend_doctors(resp.risk_score)

    # Deserialize JSON fields
    explain_ai = {}
    if resp.explainable_ai_json:
        try:
            explain_ai = json.loads(resp.explainable_ai_json)
        except Exception:
            pass

    rec_resources = []
    if resp.rec_resources_json:
        try:
            rec_resources = json.loads(resp.rec_resources_json)
        except Exception:
            pass

    return render_template(
        'result.html',
        resp=resp,
        color=color,
        recommended_doctors=recommended_doctors,
        explain_ai=explain_ai,
        rec_resources=rec_resources
    )


@app.route('/appointments', methods=['GET', 'POST'])
@login_required
def appointments():
    doctors = Doctor.query.all()
    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id')
        date      = request.form.get('date')
        time_slot = request.form.get('time_slot')
        reason    = request.form.get('reason', '')
        if not all([doctor_id, date, time_slot]):
            flash('Please fill in all appointment fields.', 'danger')
        else:
            appt = Appointment(user_id=current_user.id,
                               doctor_id=int(doctor_id),
                               date=date,
                               time_slot=time_slot,
                               reason=reason)
            db.session.add(appt)
            db.session.commit()
            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('appointments'))
    my_appointments = Appointment.query.filter_by(
        user_id=current_user.id).order_by(Appointment.booked_at.desc()).all()
    return render_template('appointments.html',
                           doctors=doctors,
                           my_appointments=my_appointments)


@app.route('/chatbot')
@login_required
def chatbot():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    using_simulated = not api_key or "your_gemini_key" in api_key
    return render_template('chatbot.html', using_simulated=using_simulated)


@app.route('/chatbot/send', methods=['POST'])
@login_required
def chatbot_send():
    print("Gemini key found:", bool(os.environ.get("GEMINI_API_KEY")))
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Please type something.'})

    # Crisis detection in chat messages (Upgrade 7)
    _, _, crisis_detected = analyze_sentiment_and_crisis(user_msg)
    if crisis_detected:
        return jsonify({
            'response': "🚨 <strong>Crisis language detected.</strong> Please know you are not alone. "
                        "If you are feeling overwhelmed, hopeless, or having thoughts of self-harm, "
                        "please connect with immediate support: call **Tele-MANAS (14416)** or "
                        "visit our Appointments tab to get immediate help. We care about your safety. 💙",
            'mode': 'crisis'
        })

    # Try Gemini API (Upgrade 9)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key and "your_gemini_key" not in api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_msg,
                config={
                    'system_instruction': (
                        "You are MindBot, a supportive mental health AI assistant. "
                        "Be empathetic, practical, and recommend seeking professional help when appropriate. "
                        "Avoid prescribing medical treatments. Respond in short, clean paragraphs using Markdown formatting. "
                        "Use positive reinforcement."
                    )
                }
            )
            bot_response = response.text
            return jsonify({'response': bot_response, 'mode': 'gemini'})
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fallback to simulated AI on error
            pass

    # Simulated AI wellness assistant fallback
    bot_response = get_simulated_ai_response(user_msg)
    return jsonify({'response': bot_response, 'mode': 'simulated'})


@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Admin access only.', 'danger')
        return redirect(url_for('dashboard'))
    total_users     = User.query.count()
    total_responses = QuestionnaireResponse.query.count()
    high_risk       = QuestionnaireResponse.query.filter_by(risk_score=2).count()
    medium_risk     = QuestionnaireResponse.query.filter_by(risk_score=1).count()
    low_risk        = QuestionnaireResponse.query.filter_by(risk_score=0).count()
    total_appts     = Appointment.query.count()
    recent_responses= QuestionnaireResponse.query.order_by(
        QuestionnaireResponse.submitted_at.desc()).limit(10).all()
    all_users       = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin.html',
                           total_users=total_users,
                           total_responses=total_responses,
                           high_risk=high_risk,
                           medium_risk=medium_risk,
                           low_risk=low_risk,
                           total_appts=total_appts,
                           recent_responses=recent_responses,
                           all_users=all_users)


# ── DB Init + Seed ───────────────────────────────────────────────────────────

def seed_doctors():
    if Doctor.query.count() == 0:
        doctors = [
            # ── Hospitals / Centers ───────────────────────────────────────
            Doctor(
                name="Karla Mind Center",
                specialization="Psychiatry & Mental Health Center",
                location="Kukatpally, Hyderabad",
                available_days="Mon – Sat",
                phone="+91 91006 39486",
                email="karlamindcenter@gmail.com",
                about="Science-led psychiatry and human-centered care. "
                      "Specializes in anxiety, depression, OCD and stress management.",
                rating=4.6,
                experience_years=8,
                tags="anxiety,depression,stress,ocd"
            ),
            Doctor(
                name="Yashoda Hospitals",
                specialization="Multi-Specialty Psychiatry Department",
                location="Malakpet / Somajiguda, Hyderabad",
                available_days="Mon – Sun (24/7 Emergency)",
                phone="+91 40 4567 4567 / +91 80659 06200",
                email=None,
                about="Multi-specialty care with a dedicated team of psychiatric doctors. "
                      "Handles complex psychiatric and neurological conditions.",
                rating=4.5,
                experience_years=15,
                tags="depression,stress,psychosis"
            ),
            Doctor(
                name="Asha Hospital",
                specialization="Psychiatry & Addiction Treatment",
                location="Banjara Hills, Hyderabad",
                available_days="Mon – Sat",
                phone="+91 96666 55558",
                email=None,
                about="Renowned for psychiatric care and addiction treatment. "
                      "One of Hyderabad's most trusted mental health facilities.",
                rating=4.8,
                experience_years=20,
                tags="addiction,depression,anxiety,stress"
            ),
            # ── Individual Psychiatrists ──────────────────────────────────
            Doctor(
                name="Dr. Sandeep Kondepi",
                specialization="Psychiatrist",
                location="Asha Hospital, Banjara Hills, Hyderabad",
                available_days="Mon, Wed, Fri",
                phone="+91 96666 55558",
                email=None,
                about="Senior Psychiatrist at Asha Hospital. "
                      "Specializes in mood disorders, anxiety and stress-related conditions.",
                rating=4.7,
                experience_years=12,
                tags="anxiety,stress,depression"
            ),
            Doctor(
                name="Dr. Srinivas K.",
                specialization="Psychiatrist",
                location="Karla Mind Center, Kukatpally, Hyderabad",
                available_days="Tue, Thu, Sat",
                phone="+91 91006 39486",
                email="karlamindcenter@gmail.com",
                about="Consultant at Karla Mind Center. "
                      "Experienced in science-led psychiatric treatment and therapy.",
                rating=4.5,
                experience_years=7,
                tags="anxiety,depression,stress,ocd"
            ),
            Doctor(
                name="Dr. Ashok K. Alimchandani",
                specialization="Psychiatrist",
                location="Apollo Health City, Jubilee Hills, Hyderabad",
                available_days="Mon – Fri",
                phone="Book via Apollo Hospitals website",
                email=None,
                about="Senior Psychiatrist at Apollo Health City. "
                      "Book appointments online via the Apollo Hospitals portal.",
                rating=4.9,
                experience_years=25,
                tags="anxiety,depression,stress,bipolar"
            ),
            Doctor(
                name="Dr. Praveen Kumar Chintapanti",
                specialization="Psychiatrist",
                location="Apollo Health City, Jubilee Hills, Hyderabad",
                available_days="Mon – Fri",
                phone="Book via Apollo Hospitals website",
                email=None,
                about="Consultant Psychiatrist at Apollo Health City. "
                      "Available for in-person and online consultations via Apollo portal.",
                rating=4.6,
                experience_years=10,
                tags="anxiety,stress,depression,therapy"
            ),
            Doctor(
                name="Dr. Mazher Ali",
                specialization="Senior Consultant Psychiatrist",
                location="CARE Hospitals, Banjara Hills, Hyderabad",
                available_days="Mon, Tue, Thu, Fri",
                phone="Contact CARE Hospitals reception",
                email=None,
                about="Senior Consultant Psychiatrist at CARE Hospitals Banjara Hills. "
                      "Extensive experience in adult psychiatry and mental health care.",
                rating=4.8,
                experience_years=18,
                tags="depression,psychosis,anxiety,stress"
            ),
        ]
        db.session.add_all(doctors)
        db.session.commit()
        print("Success: Real Hyderabad doctors seeded.")


def create_admin():
    if not User.query.filter_by(email='admin@mhsystem.com').first():
        admin_user = User(name='Admin', email='admin@mhsystem.com', is_admin=True)
        admin_user.set_password('admin123')
        db.session.add(admin_user)
        db.session.commit()
        print("Success: Admin user created: admin@mhsystem.com / admin123")


with app.app_context():
    # Automatically recreate the database if columns are outdated
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if 'questionnaire_responses' in inspector.get_table_names():
        columns = [c['name'] for c in inspector.get_columns('questionnaire_responses')]
        if 'sentiment_score' not in columns:
            print("[WARNING] Database schema is outdated. Dropping and recreating tables for mental health upgrades...")
            db.drop_all()
            db_path = os.path.join(app.instance_path, 'database.db')
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except Exception as e:
                    print(f"Could not remove DB file: {e}")
    db.create_all()
    seed_doctors()
    create_admin()

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)