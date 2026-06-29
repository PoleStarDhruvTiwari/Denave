"""
Financial Document Extractor using LangChain + LangGraph
Uses Pydantic ONLY for structured LLM output and validation
"""

import json
import re
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path

import pdfplumber
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

from config import LLM_CONFIG, QUALITY, INPUT_DIR, OUTPUT_DIR


# ═══════════════════════════════════════════
# Pydantic Models (for LLM structured output)
# ═══════════════════════════════════════════

class FinancialData(BaseModel):
    """Extracted financial data - THIS IS THE LLM OUTPUT STRUCTURE"""
    company: str = ""
    period: str = ""
    total_revenue: float = 0
    net_income: float = 0
    eps_basic: Optional[float] = None
    total_assets: float = 0
    total_liabilities: float = 0
    shareholders_equity: float = 0
    operating_cash_flow: float = 0
    free_cash_flow: float = 0
    currency: str = "USD"
    unit: str = "millions"
    
    @validator('shareholders_equity')
    def check_balance(cls, equity, values):
        """Automatic validation: A = L + E"""
        assets = values.get('total_assets', 0)
        liabilities = values.get('total_liabilities', 0)
        if assets and liabilities and equity:
            implied = assets - liabilities
            diff_pct = abs(implied - equity) / assets * 100
            if diff_pct > QUALITY['balance_diff_threshold']:
                logger.warning(
                    f"⚠️ Balance sheet mismatch: Assets({assets}) - Liabilities({liabilities}) "
                    f"= {implied} vs Equity({equity}), diff={diff_pct:.1f}%"
                )
        return equity

class RiskData(BaseModel):
    """Extracted risk factors"""
    risks: List[Dict[str, str]] = Field(default_factory=list)
    total_identified: int = 0

class SegmentData(BaseModel):
    """Extracted business segments"""
    segments: List[Dict[str, Any]] = Field(default_factory=list)


# ═══════════════════════════════════════════
# LangGraph State Definition
# ═══════════════════════════════════════════

class ExtractionState(TypedDict):
    """State that flows through the LangGraph workflow"""
    document_text: str
    sections: Dict[str, str]
    financial_data: Optional[dict]
    risk_data: Optional[dict]
    segment_data: Optional[dict]
    quality_scores: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    retry_count: int
    status: str


# ═══════════════════════════════════════════
# Document Downloader & Processor
# ═══════════════════════════════════════════

class DocumentFetcher:
    """Download financial documents from SEC EDGAR or process PDFs"""
    
    def __init__(self):
        self.headers = {"User-Agent": "Dhruv Tiwari dhruvtiwari756placement@gmail.com"}

    def from_ticker(self, ticker: str) -> str:
        """Download latest 10-K from SEC EDGAR"""
        logger.info(f"📥 Downloading 10-K for {ticker}...")

        # Get CIK
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": ticker,
            "type": "10-K",
            "count": 1,
            "output": "atom"
        }

        response = requests.get(url, params=params, headers=self.headers)

        print("Status:", response.status_code)
        print("URL:", response.url)

        soup = BeautifulSoup(response.content, "xml")

        filing_entry = soup.find("entry")
        if not filing_entry:
            raise ValueError(f"No 10-K found for {ticker}")

        filing_url = filing_entry.find("filing-href").text
        logger.info(f"Found filing: {filing_url}")

        # Open filing index page
        response = requests.get(filing_url, headers=self.headers)
        soup = BeautifulSoup(response.content, "html.parser")

        print("\n========== DOCUMENT TABLE ==========\n")

        tables = soup.find_all("table")

        for table in tables:
                for row in table.find_all("tr"):
                    cols = row.find_all("td")

                    if len(cols) < 5:
                        continue

                    description = cols[1].get_text(" ", strip=True)
                    document = cols[2].get_text(" ", strip=True)
                    doc_type = cols[3].get_text(strip=True)

                    print(f"Description : {description}")
                    print(f"Document    : {document}")
                    print(f"Type        : {doc_type}")

                    link = cols[2].find("a")
                    if link:
                        print(f"Href     : {link.get('href')}")

                    print("-" * 60)

                    # Download ONLY the real 10-K document
                    if doc_type == "10-K":
                        href = link.get("href")
                        doc_url = "https://www.sec.gov" + href.replace("/ix?doc=", "")

                        print(f"\nDownloading: {doc_url}\n")

                        doc_response = requests.get(doc_url, headers=self.headers)
                        doc_soup = BeautifulSoup(doc_response.content, "html.parser")

                        for tag in doc_soup(["script", "style"]):
                            tag.decompose()

                        text = doc_soup.get_text(separator="\n", strip=True)

                        logger.info(f"✅ Downloaded {len(text):,} characters")

                        return text

        raise ValueError("Could not find document text")

    def from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        logger.info(f"📄 Processing PDF: {pdf_path}")
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        text = '\n\n'.join(text_parts)
        logger.info(f"✅ Extracted {len(text):,} characters")
        return text
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """Split document into logical sections"""
        patterns = {
            'financials': r'(?i)Item\s+8[\.:\)]\s*Financial\s+Statements',
            'risks': r'(?i)Item\s+1A[\.:\)]\s*Risk\s+Factors',
            'business': r'(?i)Item\s+1[\.:\)]\s*Business',
            'md_and_a': r'(?i)Item\s+7[\.:\)]\s*Management.*Discussion',
        }
        
        sections = {}
        positions = []
        
        for name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                positions.append((match.start(), name))
        
        positions.sort()
        
        for i, (pos, name) in enumerate(positions):
            end = positions[i+1][0] if i+1 < len(positions) else len(text)
            sections[name] = text[pos:end][:10000]  # First 10K chars
        
        logger.info(f"Found sections: {list(sections.keys())}")
        return sections


# ═══════════════════════════════════════════
# LangGraph Extraction Workflow
# ═══════════════════════════════════════════

class FinancialExtractor:
    """
    Main extraction engine using LangChain + LangGraph
    LangGraph manages the multi-step workflow with branching
    LangChain handles LLM interactions and structured output
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
        )
        self.fetcher = DocumentFetcher()
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the extraction workflow graph
        
        Flow:
        START → Extract Financials → Validate → Quality Check
                    ↓                    ↓           ↓
                Extract Risks      Retry?     Pass/Fail
                    ↓                            ↓
                Extract Segments              END
                    ↓
                  END
        """
        workflow = StateGraph(ExtractionState)
        
        # Add nodes (each is a processing step)
        workflow.add_node("extract_financials", self._extract_financials)
        workflow.add_node("validate_data", self._validate_data)
        workflow.add_node("extract_risks", self._extract_risks)
        workflow.add_node("extract_segments", self._extract_segments)
        workflow.add_node("quality_check", self._quality_check)
        
        # Define the flow
        workflow.set_entry_point("extract_financials")
        workflow.add_edge("extract_financials", "validate_data")
        
        # Conditional: if validation fails, retry; else continue
        workflow.add_conditional_edges(
            "validate_data",
            self._should_retry,
            {
                "retry": "extract_financials",
                "continue": "extract_risks"
            }
        )
        
        workflow.add_edge("extract_risks", "extract_segments")
        workflow.add_edge("extract_segments", "quality_check")
        workflow.add_edge("quality_check", END)
        
        return workflow.compile()
    
    async def _extract_financials(self, state: ExtractionState) -> ExtractionState:
        """
        Extract financial data using LangChain with Pydantic parser
        THIS IS WHERE LANGCHAIN + PYDANTIC WORK TOGETHER:
        - Pydantic defines the exact structure we want
        - LangChain ensures LLM returns data in that structure
        """
        logger.info("💰 Extracting financial statements...")
        
        try:
            # Create parser that forces LLM to output our exact structure
            parser = PydanticOutputParser(pydantic_object=FinancialData)
            
            # Create prompt with format instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial data extraction expert.
                Extract EXACT numbers from the financial document below.
                
                {format_instructions}
                
                CRITICAL:
                - Use exact numbers as presented (no rounding)
                - Note the unit (millions/billions)
                - Negative values: include minus sign
                - If value not found, use 0 but note in warnings
                """),
                ("human", "{document}")
            ])
            
            # LangChain chain: prompt -> LLM -> structured output
            chain = prompt | self.llm | parser
            
            # Execute
            financial_section = state['sections'].get('financials', state['document_text'])
            result = await chain.ainvoke({
                "document": financial_section[:8000],
                "format_instructions": parser.get_format_instructions()
            })
            
            state['financial_data'] = result.dict()
            logger.success(f"✅ Extracted financial data for {result.period}")
            
        except Exception as e:
            logger.error(f"❌ Financial extraction failed: {str(e)}")
            state['errors'].append(str(e))
            state['retry_count'] += 1
        
        return state
    
    async def _validate_data(self, state: ExtractionState) -> ExtractionState:
        """Validate extracted data"""
        logger.info("🔍 Validating extracted data...")
        
        if not state.get('financial_data'):
            state['errors'].append("No financial data to validate")
            return state
        
        data = state['financial_data']
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Required fields present
        total_checks += 1
        required_fields = ['total_revenue', 'net_income', 'total_assets', 'total_liabilities', 'shareholders_equity']
        missing = [f for f in required_fields if not data.get(f)]
        if not missing:
            checks_passed += 1
        else:
            state['warnings'].append(f"Missing fields: {missing}")
        
        # Check 2: Balance sheet equation
        total_checks += 1
        assets = data.get('total_assets', 0)
        liabilities = data.get('total_liabilities', 0)
        equity = data.get('shareholders_equity', 0)
        if assets > 0:
            implied = assets - liabilities
            diff_pct = abs(implied - equity) / assets * 100
            if diff_pct <= QUALITY['balance_diff_threshold']:
                checks_passed += 1
            else:
                state['warnings'].append(f"Balance sheet mismatch: {diff_pct:.1f}%")
        
        # Check 3: Revenue > Net Income (usually)
        total_checks += 1
        if data.get('total_revenue', 0) >= data.get('net_income', 0):
            checks_passed += 1
        
        score = (checks_passed / total_checks * 100) if total_checks > 0 else 0
        state['quality_scores']['validation'] = score
        logger.info(f"Validation score: {score:.1f}% ({checks_passed}/{total_checks} checks passed)")
        
        return state
    
    async def _extract_risks(self, state: ExtractionState) -> ExtractionState:
        """Extract risk factors using LangChain"""
        logger.info("⚠️ Extracting risk factors...")
        
        try:
            parser = PydanticOutputParser(pydantic_object=RiskData)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract risk factors from the document.
                {format_instructions}
                
                Focus on:
                - Market risks
                - Operational risks
                - Regulatory risks
                - Financial risks
                
                Assess severity based on language used.
                """),
                ("human", "{document}")
            ])
            
            chain = prompt | self.llm | parser
            
            risk_section = state['sections'].get('risks', state['document_text'][:5000])
            result = await chain.ainvoke({
                "document": risk_section[:6000],
                "format_instructions": parser.get_format_instructions()
            })
            
            state['risk_data'] = result.dict()
            logger.success(f"✅ Extracted {result.total_identified} risks")
            
        except Exception as e:
            logger.error(f"❌ Risk extraction failed: {str(e)}")
            state['errors'].append(str(e))
        
        return state
    
    async def _extract_segments(self, state: ExtractionState) -> ExtractionState:
        """Extract business segments"""
        logger.info("🏢 Extracting business segments...")
        
        try:
            parser = PydanticOutputParser(pydantic_object=SegmentData)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract business segments from the document.
                {format_instructions}
                
                Include:
                - Segment names
                - Revenue per segment
                - Revenue percentages
                - Geographic breakdown if available
                """),
                ("human", "{document}")
            ])
            
            chain = prompt | self.llm | parser
            
            business_section = state['sections'].get('business', state['document_text'][:5000])
            result = await chain.ainvoke({
                "document": business_section[:6000],
                "format_instructions": parser.get_format_instructions()
            })
            
            state['segment_data'] = result.dict()
            logger.success(f"✅ Extracted {len(result.segments)} segments")
            
        except Exception as e:
            logger.error(f"❌ Segment extraction failed: {str(e)}")
            state['errors'].append(str(e))
        
        return state
    
    async def _quality_check(self, state: ExtractionState) -> ExtractionState:
        """Final quality assessment"""
        logger.info("📊 Running quality check...")
        
        # Calculate overall quality
        scores = state.get('quality_scores', {})
        if scores:
            overall = sum(scores.values()) / len(scores)
        else:
            overall = 0
        
        state['quality_scores']['overall'] = overall
        
        # Determine status
        if overall >= QUALITY['min_overall_score']:
            state['status'] = 'success'
            logger.success(f"✅ Quality check passed: {overall:.1f}%")
        elif overall >= QUALITY['min_dimension_score']:
            state['status'] = 'warning'
            logger.warning(f"⚠️ Quality check warning: {overall:.1f}%")
        else:
            state['status'] = 'failed'
            logger.error(f"❌ Quality check failed: {overall:.1f}%")
        
        return state
    
    def _should_retry(self, state: ExtractionState) -> str:
        """Decide whether to retry extraction"""
        if state['retry_count'] < 2 and state['errors']:
            logger.info(f"🔄 Retrying extraction (attempt {state['retry_count'] + 1})")
            return "retry"
        return "continue"
    
    async def extract(self, source: str, source_type: str = "ticker") -> Dict[str, Any]:
        """
        Main extraction method - run the complete LangGraph workflow
        
        Args:
            source: Ticker symbol or PDF path
            source_type: 'ticker' or 'pdf'
        
        Returns:
            Complete extraction results with quality scores
        """
        logger.info(f"🚀 Starting extraction for {source}")
        start_time = datetime.now()
        
        # Step 1: Get document text
        if source_type == "ticker":
            text = self.fetcher.from_ticker(source)
        elif source_type == "pdf":
            text = self.fetcher.from_pdf(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Step 2: Identify sections
        sections = self.fetcher.identify_sections(text)
        
        # Step 3: Initialize state
        initial_state: ExtractionState = {
            "document_text": text,
            "sections": sections,
            "financial_data": None,
            "risk_data": None,
            "segment_data": None,
            "quality_scores": {},
            "errors": [],
            "warnings": [],
            "retry_count": 0,
            "status": "processing"
        }
        
        # Step 4: Run LangGraph workflow
        logger.info("🔄 Running extraction workflow...")
        final_state = await self.graph.ainvoke(initial_state)
        
        # Step 5: Prepare results
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "source": source,
            "source_type": source_type,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "status": final_state['status'],
            "financial_data": final_state.get('financial_data'),
            "risk_data": final_state.get('risk_data'),
            "segment_data": final_state.get('segment_data'),
            "quality_scores": final_state['quality_scores'],
            "errors": final_state['errors'],
            "warnings": final_state['warnings'],
        }
        
        # Save result
        output_file = OUTPUT_DIR / f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.success(f"✅ Extraction complete in {duration:.1f}s")
        logger.info(f"📁 Results saved to: {output_file}")
        
        return result
    
    def extract_sync(self, source: str, source_type: str = "ticker") -> Dict[str, Any]:
        """Synchronous wrapper for extract()"""
        import asyncio
        return asyncio.run(self.extract(source, source_type))
