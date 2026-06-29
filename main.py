"""
Main entry point - Run this file to extract financial data

Usage:
    python main.py                    # Extract AAPL data
    python main.py --ticker MSFT      # Extract MSFT data
    python main.py --pdf path/to/file.pdf  # Extract from PDF
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from extractor import FinancialExtractor
from config import OPENAI_API_KEY


def display_results(result: dict):
    """Pretty print extraction results"""
    print("\n" + "="*60)
    print(f"📊 EXTRACTION RESULTS: {result['source']}")
    print("="*60)
    
    print(f"\n⏱️  Duration: {result['duration_seconds']:.2f}s")
    print(f"📈 Status: {result['status'].upper()}")
    
    # Quality scores
    if result.get('quality_scores'):
        print("\n📊 Quality Scores:")
        for metric, score in result['quality_scores'].items():
            bar = "█" * int(score / 5)
            print(f"  {metric:20s} {bar} {score:.1f}%")
    
    # Financial data
    if result.get('financial_data'):
        data = result['financial_data']
        print("\n💰 Financial Data:")
        print(f"  Company: {data.get('company', 'N/A')}")
        print(f"  Period: {data.get('period', 'N/A')}")
        print(f"  Revenue: ${data.get('total_revenue', 0):,.0f} {data.get('unit', '')}")
        print(f"  Net Income: ${data.get('net_income', 0):,.0f} {data.get('unit', '')}")
        print(f"  EPS: ${data.get('eps_basic', 0):.2f}")
        print(f"  Total Assets: ${data.get('total_assets', 0):,.0f} {data.get('unit', '')}")
        print(f"  Shareholders' Equity: ${data.get('shareholders_equity', 0):,.0f} {data.get('unit', '')}")
        print(f"  Operating Cash Flow: ${data.get('operating_cash_flow', 0):,.0f} {data.get('unit', '')}")
        print(f"  Free Cash Flow: ${data.get('free_cash_flow', 0):,.0f} {data.get('unit', '')}")
    
    # Risk data
    if result.get('risk_data'):
        risk = result['risk_data']
        print(f"\n⚠️  Risk Factors: {risk.get('total_identified', 0)} identified")
        for r in risk.get('risks', [])[:3]:
            print(f"  • [{r.get('severity', 'N/A')}] {r.get('category', 'N/A')}: {r.get('description', '')[:100]}...")
    
    # Business segments
    if result.get('segment_data'):
        seg = result['segment_data']
        print(f"\n🏢 Business Segments: {len(seg.get('segments', []))} found")
        for s in seg.get('segments', [])[:3]:
            print(f"  • {s.get('name', 'N/A')}: ${s.get('revenue', 0):,.0f} ({s.get('revenue_percentage', 0):.1f}%)")
    
    # Errors and warnings
    if result.get('errors'):
        print(f"\n❌ Errors ({len(result['errors'])}):")
        for e in result['errors']:
            print(f"  • {e}")
    
    if result.get('warnings'):
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for w in result['warnings'][:5]:
            print(f"  • {w}")
    
    print("\n" + "="*60)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Financial Document Extractor")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL, MSFT)")
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    
    args = parser.parse_args()
    
    # Validate API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        logger.error("❌ Please set your OPENAI_API_KEY in the .env file")
        sys.exit(1)
    
    # Initialize extractor
    extractor = FinancialExtractor()
    
    # Determine source
    if args.pdf:
        source = args.pdf
        source_type = "pdf"
    elif args.ticker:
        source = args.ticker.upper()
        source_type = "ticker"
    else:
        # Default: Extract Apple data
        logger.info("No arguments provided, defaulting to AAPL")
        source = "AAPL"
        source_type = "ticker"
    
    # Run extraction
    try:
        result = extractor.extract_sync(source, source_type)
        display_results(result)
    except KeyboardInterrupt:
        logger.warning("⚠️ Extraction interrupted by user")
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()