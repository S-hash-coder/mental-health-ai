@echo off
echo =======================================
echo  MindCare AI - Setup Script
echo =======================================
echo.
echo [1/3] Installing Python dependencies...
pip install flask flask_sqlalchemy flask_login pandas numpy scikit-learn werkzeug
echo.
echo [2/3] Training the ML model...
python train_model.py
echo.
echo [3/3] Starting Flask server...
echo  Open browser: http://127.0.0.1:5000
echo  Admin login:  admin@mhsystem.com / admin123
echo.
python app.py
