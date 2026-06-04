"""
app.py — Main Flask Application
Mental Health AI Support System
"""

import os
import pickle
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

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
    risk_level      = db.Column(db.String(20))   # Low / Medium / High
    risk_score      = db.Column(db.Integer)       # 0, 1, 2
    recommendation  = db.Column(db.Text)
    submitted_at    = db.Column(db.DateTime, default=datetime.utcnow)


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

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
RISK_COLORS = {0: 'success', 1: 'warning', 2: 'danger'}

RECOMMENDATIONS = {
    0: ("Great job taking care of yourself! Keep up the healthy routine. "
        "Maintain regular sleep, exercise, and social connections."),
    1: ("Your responses indicate moderate stress. Consider trying mindfulness, "
        "talking to a friend, or scheduling a counseling session to help manage your wellbeing."),
    2: ("Your responses indicate high stress and risk. We strongly recommend speaking "
        "with a counselor or mental health professional as soon as possible. You are not alone — help is available.")
}

CHATBOT_RULES = [
    (["stress", "stressed", "overwhelmed"],
     "I hear you. Try the 4-7-8 breathing technique: inhale for 4s, hold for 7s, exhale for 8s. 💙"),
    (["sad", "depressed", "unhappy", "hopeless"],
     "I'm sorry you're feeling this way. Please talk to someone you trust or consider speaking with a counselor. You matter."),
    (["sleep", "insomnia", "tired", "exhausted"],
     "Good sleep is crucial. Try a consistent bedtime, no screens 1hr before bed, and a calm environment."),
    (["anxious", "anxiety", "nervous", "panic"],
     "Anxiety can feel overwhelming. Grounding yourself with the 5-4-3-2-1 technique can help: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."),
    (["study", "exam", "pressure", "fail"],
     "Academic pressure is real. Break your work into small tasks, take regular breaks, and remember: your worth isn't defined by grades."),
    (["lonely", "alone", "isolated"],
     "Loneliness is painful. Try reaching out to one person today — even a short message can help. Community matters."),
    (["help", "support", "counselor", "doctor"],
     "Our appointment system can connect you with a counselor. Head to the Appointments page to book a session."),
    (["hi", "hello", "hey"],
     "Hello! 👋 I'm MindBot, your mental health support companion. How are you feeling today?"),
]

def get_chatbot_response(user_input):
    text = user_input.lower()
    for keywords, response in CHATBOT_RULES:
        if any(kw in text for kw in keywords):
            return response
    return ("Thank you for sharing. I'm here to listen. For serious concerns, "
            "please consider booking an appointment with one of our counselors. 💙")


def predict_risk(sleep, stress, pressure, anxiety, mood):
    if model is None:
        # Fallback simple rule if model not trained yet
        score = (stress + pressure + anxiety) / 3
        if sleep < 5 or score >= 7:
            return 2, RISK_LABELS[2], RECOMMENDATIONS[2]
        elif sleep < 7 or score >= 5:
            return 1, RISK_LABELS[1], RECOMMENDATIONS[1]
        return 0, RISK_LABELS[0], RECOMMENDATIONS[0]
    features = [[float(sleep), float(stress), float(pressure), float(anxiety), float(mood)]]
    pred = int(model.predict(features)[0])
    return pred, RISK_LABELS[pred], RECOMMENDATIONS[pred]


def recommend_doctors(risk_score, user_location=None):
    doctors = Doctor.query.all()

    scored = []

    for doc in doctors:
        score = 0

        # 1. Risk matching
        if risk_score == 2:  # High stress
            if "anxiety" in (doc.tags or ""):
                score += 3
            if "depression" in (doc.tags or ""):
                score += 3
        elif risk_score == 1:
            score += 2
        else:
            score += 1

        # 2. Rating weight
        score += (doc.rating or 4) * 1.5

        # 3. Experience weight
        score += (doc.experience_years or 5) * 0.5

        scored.append((score, doc))

    # sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored[:3]]


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

            score, label, recommendation = predict_risk(sleep, stress, pressure, anxiety, mood)

            resp = QuestionnaireResponse(
                user_id=current_user.id,
                sleep_hours=sleep,
                stress_level=stress,
                study_pressure=pressure,
                social_anxiety=anxiety,
                mood_score=mood,
                feelings_text=feelings,
                risk_level=label,
                risk_score=score,
                recommendation=recommendation
            )
            db.session.add(resp)
            db.session.commit()
            return redirect(url_for('result', response_id=resp.id))
        except (KeyError, ValueError) as e:
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

    return render_template(
        'result.html',
        resp=resp,
        color=color,
        recommended_doctors=recommended_doctors
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
    return render_template('chatbot.html')


@app.route('/chatbot/send', methods=['POST'])
@login_required
def chatbot_send():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Please type something.'})
    bot_response = get_chatbot_response(user_msg)
    return jsonify({'response': bot_response})


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
    db.create_all()
    seed_doctors()
    create_admin()

if __name__ == '__main__':
    app.run(debug=True)
