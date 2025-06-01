"""
Data models for AI-Powered Personal Financial Planning MCP Server
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentGoal(Enum):
    RETIREMENT = "retirement"
    EDUCATION = "education"
    HOME_PURCHASE = "home_purchase"
    EMERGENCY_FUND = "emergency_fund"
    WEALTH_BUILDING = "wealth_building"


@dataclass
class FinancialProfile:
    user_id: str
    age: int
    annual_income: float
    monthly_expenses: Dict[str, float]
    current_savings: float
    current_debt: float
    risk_tolerance: RiskTolerance
    investment_goals: List[Dict[str, Any]]  # goal_type, target_amount, timeline
    dependents: int
    employment_stability: str  # stable, moderate, unstable
    created_at: datetime
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['risk_tolerance'] = self.risk_tolerance.value
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialProfile':
        """Create from dictionary"""
        data['risk_tolerance'] = RiskTolerance(data['risk_tolerance'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class MarketConditions:
    market_trend: str  # bullish, bearish, sideways
    volatility_index: float
    interest_rates: Dict[str, float]
    inflation_rate: float
    sector_performance: Dict[str, float]
    economic_indicators: Dict[str, float]
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketConditions':
        """Create from dictionary"""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class BudgetPlan:
    monthly_income: float
    expense_categories: Dict[str, float]
    savings_rate: float
    emergency_fund_target: float
    discretionary_spending: float
    budget_recommendations: List[str]
    savings_timeline: Dict[str, int]  # months to reach various savings goals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class InvestmentRecommendation:
    asset_allocation: Dict[str, float]  # asset_class -> percentage
    specific_investments: List[Dict[str, Any]]
    risk_score: float
    expected_return: float
    time_horizon: int
    rebalancing_frequency: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class FinancialPlan:
    user_id: str
    financial_profile: FinancialProfile
    budget_plan: BudgetPlan
    investment_recommendations: List[InvestmentRecommendation]
    market_conditions: MarketConditions
    plan_summary: str
    action_items: List[str]
    review_schedule: str
    confidence_score: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'financial_profile': self.financial_profile.to_dict(),
            'budget_plan': self.budget_plan.to_dict(),
            'investment_recommendations': [rec.to_dict() for rec in self.investment_recommendations],
            'market_conditions': self.market_conditions.to_dict(),
            'plan_summary': self.plan_summary,
            'action_items': self.action_items,
            'review_schedule': self.review_schedule,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class AIAnalysisRequest:
    """Request structure for AI analysis"""
    user_profile: Dict[str, Any]
    market_data: Optional[Dict[str, Any]] = None
    analysis_type: str = "comprehensive"
    include_recommendations: bool = True


@dataclass
class AIAnalysisResponse:
    """Response structure for AI analysis"""
    analysis_summary: str
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data
