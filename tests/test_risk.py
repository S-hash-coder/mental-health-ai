import unittest
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app import (
    analyze_sentiment_and_crisis, 
    calculate_hybrid_risk_and_explain, 
    recommend_doctors,
    app,
    db,
    Doctor
)

class TestMentalHealthRiskModel(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_sentiment_and_crisis_detection(self):
        # Test crisis detection
        comp, label, crisis = analyze_sentiment_and_crisis("I want to end my life and disappear")
        self.assertTrue(crisis)
        self.assertLess(comp, 0)
        
        # Test positive sentiment
        comp, label, crisis = analyze_sentiment_and_crisis("I feel absolutely amazing and happy today")
        self.assertFalse(crisis)
        self.assertGreater(comp, 0.4)
        self.assertEqual(label, "Excellent / Calm")
        
        # Test empty input
        comp, label, crisis = analyze_sentiment_and_crisis("")
        self.assertFalse(crisis)
        self.assertEqual(comp, 0.0)
        self.assertEqual(label, "Neutral / Reflective")

    def test_hybrid_risk_calculations(self):
        # 1. Low risk inputs
        res = calculate_hybrid_risk_and_explain(
            sleep=8.0, stress=2, pressure=1, anxiety=2, mood=9, feelings_text="Feeling peaceful and calm."
        )
        self.assertEqual(res['final_level'], "Low")
        self.assertEqual(res['final_score'], 0)
        self.assertFalse(res['crisis_detected'])
        self.assertEqual(res['sleep_goal'], 7.5)
        
        # 2. High risk inputs
        res = calculate_hybrid_risk_and_explain(
            sleep=4.0, stress=8, pressure=7, anxiety=8, mood=3, feelings_text="I'm feeling very sad, exhausted, and overwhelmed."
        )
        self.assertEqual(res['final_level'], "High")
        self.assertEqual(res['final_score'], 2)
        self.assertFalse(res['crisis_detected'])
        
        # 3. Crisis override inputs (Critical)
        res = calculate_hybrid_risk_and_explain(
            sleep=7.0, stress=5, pressure=4, anxiety=3, mood=6, feelings_text="I want to kill myself."
        )
        self.assertEqual(res['final_level'], "Critical")
        self.assertEqual(res['final_score'], 3)
        self.assertTrue(res['crisis_detected'])
        self.assertEqual(res['hybrid_index'], 95.0)

    def test_explainable_ai_mapping(self):
        # High stress contribution
        res = calculate_hybrid_risk_and_explain(
            sleep=8.0, stress=9, pressure=1, anxiety=1, mood=9, feelings_text=""
        )
        explain = res['explainable_ai']
        self.assertIn("High Stress", explain)
        self.assertNotIn("Poor Sleep", explain)
        
        # Poor sleep contribution
        res = calculate_hybrid_risk_and_explain(
            sleep=3.0, stress=2, pressure=1, anxiety=1, mood=9, feelings_text=""
        )
        explain = res['explainable_ai']
        self.assertIn("Poor Sleep", explain)
        
    def test_resource_recommendation_engine(self):
        # Matches sleep resources
        res = calculate_hybrid_risk_and_explain(
            sleep=4.0, stress=2, pressure=2, anxiety=2, mood=8, feelings_text="I am tired and can't sleep"
        )
        titles = [r['title'] for r in res['rec_resources']]
        self.assertIn("Sleep Hygiene Guidelines", titles)
        
        # Matches study stress resources
        res = calculate_hybrid_risk_and_explain(
            sleep=8.0, stress=7, pressure=8, anxiety=2, mood=8, feelings_text="exams and study are stressing me out"
        )
        titles = [r['title'] for r in res['rec_resources']]
        self.assertIn("Time Management Tool: Pomodoro Method", titles)

if __name__ == '__main__':
    unittest.main()
