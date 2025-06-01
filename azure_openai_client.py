"""
Azure OpenAI client for AI-powered financial analysis
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential

from models import AIAnalysisRequest, AIAnalysisResponse, FinancialProfile, MarketConditions

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Azure OpenAI client for financial analysis"""

    def __init__(self):
        self.client = None
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        try:
            # Option 1: Using API key
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            if api_key and endpoint:
                self.client = AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=self.api_version,
                    azure_endpoint=endpoint
                )
                logger.info("Azure OpenAI client initialized with API key")
            else:
                # Option 2: Using Azure credential (managed identity, etc.)
                credential = DefaultAzureCredential()
                self.client = AsyncAzureOpenAI(
                    azure_ad_token_provider=credential.get_token,
                    api_version=self.api_version,
                    azure_endpoint=endpoint
                )
                logger.info("Azure OpenAI client initialized with Azure credential")
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    async def analyze_financial_profile(self, request: AIAnalysisRequest) -> AIAnalysisResponse:
        """Analyze financial profile using Azure OpenAI"""
        
        system_prompt = self._get_financial_analyst_system_prompt()
        user_prompt = self._create_profile_analysis_prompt(request)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            return AIAnalysisResponse(
                analysis_summary=analysis_data.get("summary", ""),
                insights=analysis_data.get("insights", []),
                recommendations=analysis_data.get("recommendations", []),
                confidence_score=analysis_data.get("confidence_score", 0.8),
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in financial profile analysis: {e}")
            # Return fallback analysis
            return AIAnalysisResponse(
                analysis_summary="Unable to generate AI analysis at this time",
                insights=["Manual analysis recommended"],
                recommendations=["Consult with financial advisor"],
                confidence_score=0.0,
                analysis_timestamp=datetime.now()
            )

    async def generate_investment_strategy(self, profile: FinancialProfile, 
                                         market_conditions: MarketConditions) -> Dict[str, Any]:
        """Generate investment strategy using Azure OpenAI"""
        
        system_prompt = self._get_investment_advisor_system_prompt()
        user_prompt = self._create_investment_strategy_prompt(profile, market_conditions)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in investment strategy generation: {e}")
            return self._get_fallback_investment_strategy()

    async def analyze_market_sentiment(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """Analyze market sentiment and provide insights"""
        
        system_prompt = self._get_market_analyst_system_prompt()
        user_prompt = self._create_market_analysis_prompt(market_conditions)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {e}")
            return {"sentiment": "neutral", "insights": [], "confidence": 0.5}

    async def generate_budget_optimization(self, profile: FinancialProfile) -> Dict[str, Any]:
        """Generate budget optimization suggestions"""
        
        system_prompt = self._get_budget_advisor_system_prompt()
        user_prompt = self._create_budget_optimization_prompt(profile)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in budget optimization: {e}")
            return {"recommendations": [], "optimizations": {}, "savings_potential": 0}

    def _get_financial_analyst_system_prompt(self) -> str:
        """System prompt for financial analysis"""
        return """You are an expert financial analyst specializing in personal finance. 
        Analyze the provided financial profile and provide comprehensive insights.
        
        Your response must be valid JSON with the following structure:
        {
            "summary": "Brief summary of financial health",
            "insights": ["List of key insights"],
            "recommendations": ["List of actionable recommendations"],
            "confidence_score": 0.8
        }
        
        Focus on:
        - Financial health assessment
        - Risk analysis
        - Goal alignment
        - Actionable recommendations
        """

    def _get_investment_advisor_system_prompt(self) -> str:
        """System prompt for investment advice"""
        return """You are a certified investment advisor with expertise in portfolio management.
        Create personalized investment strategies based on user profiles and market conditions.
        
        Your response must be valid JSON with the following structure:
        {
            "asset_allocation": {"stocks": 0.7, "bonds": 0.25, "cash": 0.05},
            "investment_products": [{"name": "Product", "allocation": 0.3, "rationale": "Why"}],
            "strategy_summary": "Overall strategy explanation",
            "risk_assessment": "Risk level and justification",
            "rebalancing_schedule": "When to rebalance",
            "expected_return": 0.08
        }
        
        Consider:
        - Risk tolerance and capacity
        - Time horizon
        - Market conditions
        - Diversification principles
        - Cost efficiency
        """

    def _get_market_analyst_system_prompt(self) -> str:
        """System prompt for market analysis"""
        return """You are a market research analyst specializing in economic trends and market sentiment.
        Analyze current market conditions and provide actionable insights.
        
        Your response must be valid JSON with the following structure:
        {
            "sentiment": "bullish/bearish/neutral",
            "key_trends": ["List of major trends"],
            "risk_factors": ["List of risk factors"],
            "opportunities": ["List of opportunities"],
            "outlook": "Short-term market outlook",
            "confidence": 0.8
        }
        
        Focus on:
        - Market sentiment analysis
        - Economic indicators interpretation
        - Sector performance analysis
        - Risk assessment
        """

    def _get_budget_advisor_system_prompt(self) -> str:
        """System prompt for budget optimization"""
        return """You are a personal finance coach specializing in budget optimization.
        Analyze spending patterns and provide optimization recommendations.
        
        Your response must be valid JSON with the following structure:
        {
            "budget_health_score": 75,
            "spending_analysis": {"category": "analysis"},
            "optimization_opportunities": [{"category": "housing", "potential_savings": 200, "method": "refinance"}],
            "recommendations": ["List of actionable recommendations"],
            "priority_actions": ["Most important actions to take"]
        }
        
        Focus on:
        - Spending pattern analysis
        - Cost reduction opportunities
        - Savings maximization
        - Emergency fund building
        """

    def _create_profile_analysis_prompt(self, request: AIAnalysisRequest) -> str:
        """Create prompt for profile analysis"""
        profile_data = request.user_profile
        
        prompt = f"""
        Analyze the following financial profile:
        
        Personal Information:
        - Age: {profile_data.get('age', 'N/A')}
        - Annual Income: ${profile_data.get('annual_income', 0):,.2f}
        - Dependents: {profile_data.get('dependents', 0)}
        - Employment Stability: {profile_data.get('employment_stability', 'N/A')}
        
        Financial Situation:
        - Current Savings: ${profile_data.get('current_savings', 0):,.2f}
        - Current Debt: ${profile_data.get('current_debt', 0):,.2f}
        - Monthly Expenses: {profile_data.get('monthly_expenses', {})}
        - Risk Tolerance: {profile_data.get('risk_tolerance', 'N/A')}
        
        Investment Goals: {profile_data.get('investment_goals', [])}
        
        Provide a comprehensive financial analysis with specific, actionable recommendations.
        """
        
        return prompt

    def _create_investment_strategy_prompt(self, profile: FinancialProfile, 
                                         market_conditions: MarketConditions) -> str:
        """Create prompt for investment strategy"""
        
        prompt = f"""
        Create an investment strategy for the following profile:
        
        Client Profile:
        - Age: {profile.age}
        - Annual Income: ${profile.annual_income:,.2f}
        - Risk Tolerance: {profile.risk_tolerance.value}
        - Investment Goals: {profile.investment_goals}
        - Current Savings: ${profile.current_savings:,.2f}
        
        Current Market Conditions:
        - Market Trend: {market_conditions.market_trend}
        - Volatility Index: {market_conditions.volatility_index}
        - Interest Rates: {market_conditions.interest_rates}
        - Inflation Rate: {market_conditions.inflation_rate}%
        
        Create a personalized investment strategy with specific asset allocation and product recommendations.
        Consider the current market environment and adjust the strategy accordingly.
        """
        
        return prompt

    def _create_market_analysis_prompt(self, market_conditions: MarketConditions) -> str:
        """Create prompt for market analysis"""
        
        prompt = f"""
        Analyze the current market conditions:
        
        Market Overview:
        - Trend: {market_conditions.market_trend}
        - Volatility Index: {market_conditions.volatility_index}
        - Inflation Rate: {market_conditions.inflation_rate}%
        
        Interest Rates: {market_conditions.interest_rates}
        
        Sector Performance: {market_conditions.sector_performance}
        
        Economic Indicators: {market_conditions.economic_indicators}
        
        Provide insights on market sentiment, key trends, risks, and opportunities for investors.
        """
        
        return prompt

    def _create_budget_optimization_prompt(self, profile: FinancialProfile) -> str:
        """Create prompt for budget optimization"""
        
        monthly_income = profile.annual_income / 12
        total_expenses = sum(profile.monthly_expenses.values())
        
        prompt = f"""
        Optimize the budget for the following financial profile:
        
        Income and Expenses:
        - Monthly Income: ${monthly_income:,.2f}
        - Monthly Expenses: {profile.monthly_expenses}
        - Total Monthly Expenses: ${total_expenses:,.2f}
        - Current Savings Rate: {((monthly_income - total_expenses) / monthly_income * 100):.1f}%
        
        Personal Situation:
        - Age: {profile.age}
        - Dependents: {profile.dependents}
        - Current Debt: ${profile.current_debt:,.2f}
        
        Analyze spending patterns and provide specific optimization recommendations to improve savings rate and financial health.
        """
        
        return prompt

    def _get_fallback_investment_strategy(self) -> Dict[str, Any]:
        """Fallback investment strategy when AI is unavailable"""
        return {
            "asset_allocation": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
            "investment_products": [
                {"name": "Total Stock Market Index", "allocation": 0.6, "rationale": "Broad market exposure"}
            ],
            "strategy_summary": "Conservative balanced portfolio",
            "risk_assessment": "Moderate risk",
            "rebalancing_schedule": "Quarterly",
            "expected_return": 0.07
        }
