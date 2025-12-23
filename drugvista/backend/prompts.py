"""
Prompt Templates for Multi-step Reasoning
Each prompt is designed for specific reasoning steps
"""

class PromptTemplates:
    
    context_analysis = """
You are a biomedical AI analyzing a query about drugs, diseases, or molecules.

QUERY: {query}

RETRIEVED CONTEXT:
{retrieved_context}

TASK: Analyze the query and context to understand:
1. What type of analysis is needed (disease, molecule, clinical trial, etc.)
2. Key entities mentioned (drug names, diseases, molecular targets)
3. The user's likely intent (research, clinical decision, market analysis)

Provide a clear, factual analysis focusing on what can be determined from the retrieved documents.
Avoid speculation. If information is missing, state that clearly.

ANALYSIS:
"""

    clinical_analysis = """
You are a clinical research expert analyzing biomedical evidence.

QUERY: {query}
CONTEXT UNDERSTANDING: {context_understanding}

CLINICAL EVIDENCE:
{clinical_evidence}

TASK: Provide clinical reasoning based ONLY on the evidence provided:
1. Assess clinical viability based on available data
2. Identify potential risks or safety concerns mentioned
3. Note any clinical trial results or outcomes
4. Highlight mechanism of action if described

Be precise and cite specific evidence. Do not hallucinate information not present in the documents.
If evidence is insufficient, state that clearly.

CLINICAL ANALYSIS:
"""

    market_analysis = """
You are a pharmaceutical market analyst reviewing market intelligence.

QUERY: {query}
CONTEXT UNDERSTANDING: {context_understanding}

MARKET INTELLIGENCE:
{market_intelligence}

TASK: Analyze market factors based on provided information:
1. Market demand signals mentioned in the documents
2. Competitive landscape insights
3. Regulatory or approval status if mentioned
4. Commercial viability indicators

Base your analysis strictly on the provided market intelligence documents.
Do not speculate beyond what is explicitly stated.

MARKET ANALYSIS:
"""

    decision_synthesis = """
You are a pharmaceutical decision-making AI synthesizing multiple analyses.

QUERY: {query}

CONTEXT ANALYSIS:
{context_analysis}

CLINICAL ANALYSIS:
{clinical_analysis}

MARKET ANALYSIS:
{market_analysis}

TASK: Synthesize a final recommendation by:
1. Weighing clinical evidence against market factors
2. Identifying the most critical decision factors
3. Assessing overall risk-benefit profile
4. Providing a clear recommendation with reasoning

Your synthesis should be balanced, evidence-based, and acknowledge limitations.
Focus on actionable insights for pharmaceutical decision-makers.

DECISION SYNTHESIS:
"""