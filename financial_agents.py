"""
AI Financial Agents for specialized financial analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from models import FinancialProfile, MarketConditions, BudgetPlan, InvestmentRecommendation
from azure_openai_client import AzureOpenAIClient

logger = logging.getLogger(__name__)


class AIFinancialAgent:
    """Base class for specialized AI agents"""
    
    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.context_history = []
        self.ai_client = AzureOpenAIClient()
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override in specialized agents"""
        raise NotImplementedError


class FinancialProfileAnalyzer(AIFinancialAgent):
    """Agent specialized in analyzing user financial profiles"""
    
    def __init__(self):
        super().__init__("ProfileAnalyzer", "Financial Profile Analysis")
    
    async def analyze(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's financial profile and extract insights"""
        
        # Calculate key financial ratios
        annual_income = profile_data.get('annual_income', 0)
        monthly_income = annual_income / 12
        total_monthly_expenses = sum(profile_data.get('monthly_expenses', {}).values())
        savings_rate = (monthly_income - total_monthly_expenses) / monthly_income if monthly_income > 0 else 0
        
        # Debt-to-income ratio
        monthly_debt = profile_data.get('current_debt', 0) / 12  # Assume annual debt payments
        debt_to_income = monthly_debt / monthly_income if monthly_income > 0 else 0
        
        # Emergency fund adequacy
        current_savings = profile_data.get('current_savings', 0)
        emergency_fund_months = current_savings / total_monthly_expenses if total_monthly_expenses > 0 else 0
        
        # Risk capacity assessment
        age = profile_data.get('age', 30)
        dependents = profile_data.get('dependents', 0)
        employment_stability = profile_data.get('employment_stability', 'moderate')
        
        risk_capacity_score = self._calculate_risk_capacity(age, dependents, employment_stability, savings_rate)
        
        # Basic analysis
        basic_analysis = {
            'financial_health_score': self._calculate_financial_health_score(
                savings_rate, debt_to_income, emergency_fund_months
            ),
            'savings_rate': round(savings_rate * 100, 2),
            'debt_to_income_ratio': round(debt_to_income * 100, 2),
            'emergency_fund_months': round(emergency_fund_months, 1),
            'risk_capacity_score': risk_capacity_score,
            'spending_breakdown': self._analyze_spending_patterns(profile_data.get('monthly_expenses', {})),
            'financial_priorities': self._identify_priorities(profile_data),
            'improvement_areas': self._identify_improvement_areas(savings_rate, debt_to_income, emergency_fund_months)
        }
        
        # Enhanced AI analysis
        try:
            from models import AIAnalysisRequest
            ai_request = AIAnalysisRequest(
                user_profile=profile_data,
                analysis_type="financial_profile"
            )
            ai_analysis = await self.ai_client.analyze_financial_profile(ai_request)
            
            # Combine basic and AI analysis
            basic_analysis.update({
                'ai_summary': ai_analysis.analysis_summary,
                'ai_insights': ai_analysis.insights,
                'ai_recommendations': ai_analysis.recommendations,
                'ai_confidence': ai_analysis.confidence_score
            })
            
        except Exception as e:
            logger.warning(f"AI analysis failed, using basic analysis: {e}")
        
        return basic_analysis
    
    def _calculate_financial_health_score(self, savings_rate: float, debt_to_income: float, emergency_months: float) -> int:
        """Calculate overall financial health score (0-100)"""
        score = 0
        
        # Savings rate component (40 points max)
        if savings_rate >= 0.20:
            score += 40
        elif savings_rate >= 0.15:
            score += 30
        elif savings_rate >= 0.10:
            score += 20
        elif savings_rate >= 0.05:
            score += 10
        
        # Debt-to-income component (30 points max)
        if debt_to_income <= 0.10:
            score += 30
        elif debt_to_income <= 0.20:
            score += 20
        elif debt_to_income <= 0.30:
            score += 10
        
        # Emergency fund component (30 points max)
        if emergency_months >= 6:
            score += 30
        elif emergency_months >= 3:
            score += 20
        elif emergency_months >= 1:
            score += 10
        
        return min(score, 100)
    
    def _calculate_risk_capacity(self, age: int, dependents: int, employment: str, savings_rate: float) -> float:
        """Calculate risk capacity score (0-1)"""
        base_score = 0.5
        
        # Age factor (younger = higher capacity)
        age_factor = max(0, (65 - age) / 40)
        
        # Dependents factor
        dependents_factor = max(0, 1 - (dependents * 0.15))
        
        # Employment stability factor
        employment_factors = {'stable': 1.0, 'moderate': 0.8, 'unstable': 0.5}
        employment_factor = employment_factors.get(employment, 0.8)
        
        # Savings rate factor
        savings_factor = min(1.0, savings_rate * 2)
        
        risk_capacity = (age_factor * 0.3 + dependents_factor * 0.2 + 
                        employment_factor * 0.3 + savings_factor * 0.2)
        
        return min(1.0, max(0.1, risk_capacity))
    
    def _analyze_spending_patterns(self, expenses: Dict[str, float]) -> Dict[str, Any]:
        """Analyze spending patterns and categorize"""
        total_expenses = sum(expenses.values())
        if total_expenses == 0:
            return {}
        
        categories = {
            'essential': ['housing', 'utilities', 'groceries', 'insurance', 'transportation'],
            'lifestyle': ['dining', 'entertainment', 'shopping', 'subscriptions'],
            'financial': ['debt_payments', 'savings', 'investments']
        }
        
        breakdown = {}
        for category, expense_types in categories.items():
            category_total = sum(expenses.get(expense_type, 0) for expense_type in expense_types)
            breakdown[category] = {
                'amount': category_total,
                'percentage': round((category_total / total_expenses) * 100, 1)
            }
        
        return breakdown
    
    def _identify_priorities(self, profile_data: Dict[str, Any]) -> List[str]:
        """Identify financial priorities based on profile"""
        priorities = []
        
        age = profile_data.get('age', 30)
        current_savings = profile_data.get('current_savings', 0)
        monthly_expenses = sum(profile_data.get('monthly_expenses', {}).values())
        goals = profile_data.get('investment_goals', [])
        
        # Emergency fund priority
        if current_savings < monthly_expenses * 3:
            priorities.append("Build emergency fund (3-6 months expenses)")
        
        # Debt reduction priority
        if profile_data.get('current_debt', 0) > 0:
            priorities.append("Reduce high-interest debt")
        
        # Retirement planning
        if age > 25 and not any(goal.get('goal_type') == 'retirement' for goal in goals):
            priorities.append("Start retirement planning")
        
        # Goal-specific priorities
        for goal in goals:
            if goal.get('timeline', 0) <= 5:
                priorities.append(f"Short-term goal: {goal.get('goal_type', 'Unknown')}")
        
        return priorities
    
    def _identify_improvement_areas(self, savings_rate: float, debt_to_income: float, emergency_months: float) -> List[str]:
        """Identify areas for financial improvement"""
        improvements = []
        
        if savings_rate < 0.10:
            improvements.append("Increase savings rate to at least 10%")
        
        if debt_to_income > 0.30:
            improvements.append("Reduce debt-to-income ratio below 30%")
        
        if emergency_months < 3:
            improvements.append("Build emergency fund to 3-6 months of expenses")
        
        return improvements


class MarketDataAgent(AIFinancialAgent):
    """Agent specialized in fetching and analyzing market data"""
    
    def __init__(self):
        super().__init__("MarketAnal
