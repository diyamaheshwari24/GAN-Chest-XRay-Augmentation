"""
LLM Module
Falcon-7B based report generation for chest X-ray interpretations
"""

from .report_generator import generate_reports, generate_report_offline, make_prompt

__all__ = ['generate_reports', 'generate_report_offline', 'make_prompt']
