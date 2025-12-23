"""
RAG Pipeline with Multi-step Reasoning
AWS Mapping: Orchestrates Bedrock (LLM) + SageMaker (embeddings) + S3 (documents)
"""
import json
import logging
from typing import Dict, List
from vector_store import VectorStore
from prompts import PromptTemplates
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline components"""
        # AWS Bedrock would replace OpenAI client
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        # Load vector store (AWS: S3 + SageMaker embeddings)
        self.vector_store = VectorStore()
        self.prompts = PromptTemplates()
        
        logger.info("RAG Pipeline initialized")
    
    def analyze(self, query: str) -> Dict:
        """
        Main analysis pipeline with multi-step reasoning
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.vector_store.search(query, top_k=5)
            
            # Check if we have relevant documents (similarity threshold)
            relevant_docs = [doc for doc in retrieved_docs if doc.get('similarity_score', 0) > 0.3]
            has_relevant_data = len(relevant_docs) >= 1
            
            if has_relevant_data:
                # Use RAG with retrieved documents
                context_analysis = self._analyze_context(query, relevant_docs)
                clinical_analysis = self._analyze_clinical(query, relevant_docs, context_analysis)
                market_analysis = self._analyze_market(query, relevant_docs, context_analysis)
                final_decision = self._synthesize_decision(query, context_analysis, clinical_analysis, market_analysis)
                
                return self._format_response(
                    query, relevant_docs, context_analysis, 
                    clinical_analysis, market_analysis, final_decision,
                    from_knowledge_base=True
                )
            else:
                # No relevant data - use OpenAI general knowledge with low confidence
                logger.info(f"No relevant documents found for query: {query[:50]}. Using general knowledge.")
                return self._generate_general_insights(query)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._fallback_response(str(e))
    
    def _generate_general_insights(self, query: str) -> Dict:
        """Generate insights using OpenAI's general knowledge when no relevant docs exist"""
        prompt = f"""You are a pharmaceutical research assistant. The user is asking about a topic 
that is NOT in our internal knowledge base. Provide helpful general insights based on your training data.

Query: {query}

Please provide:
1. A brief clinical assessment based on general medical knowledge
2. Known risks or concerns in this area
3. General market outlook if applicable
4. A cautious recommendation

Be clear that this is based on general knowledge, not proprietary data.
Format your response clearly with sections."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            content = response.choices[0].message.content
            
            # Parse general insights
            viability = "Medium"
            if any(word in content.lower() for word in ['promising', 'effective', 'approved']):
                viability = "Medium"  # Keep medium even if positive since it's general knowledge
            elif any(word in content.lower() for word in ['failed', 'dangerous', 'withdrawn']):
                viability = "Low"
            
            risks = ["Limited proprietary data available"]
            if 'risk' in content.lower() or 'concern' in content.lower():
                risks.append("See detailed analysis for specific risks")
            
            market_signal = "Unknown"
            if any(word in content.lower() for word in ['growing', 'demand', 'billion']):
                market_signal = "Moderate"
            
            return {
                "clinical_viability": viability,
                "key_evidence": ["⚠️ Based on general AI knowledge (not internal data)"],
                "major_risks": risks,
                "market_signal": market_signal,
                "recommendation": "Investigate Further",
                "confidence_score": 0.25,  # Low confidence for general knowledge
                "explanation": f"""
**⚠️ Note**: No relevant documents found in the knowledge base for this query. 
The following insights are based on general AI knowledge and should be verified with authoritative sources.

**Query**: {query}

**General Insights**:
{content}

**Confidence**: LOW (25%) - This analysis is not based on proprietary data. 
Please upload relevant documents or consult domain experts for higher confidence analysis.
                """.strip()
            }
            
        except Exception as e:
            logger.error(f"General insights generation failed: {e}")
            return self._fallback_response(str(e))
    
    def _analyze_context(self, query: str, docs: List[Dict]) -> Dict:
        """Step 1: Context Understanding"""
        context = "\n".join([f"Doc {i+1}: {doc['content'][:500]}" for i, doc in enumerate(docs)])
        
        prompt = self.prompts.context_analysis.format(
            query=query,
            retrieved_context=context
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return {"analysis": content, "docs_used": len(docs)}
            
        except Exception as e:
            logger.warning(f"Context analysis failed: {e}")
            return {"analysis": f"Basic analysis of query: {query}", "docs_used": 0}
    
    def _analyze_clinical(self, query: str, docs: List[Dict], context: Dict) -> Dict:
        """Step 2: Clinical Reasoning"""
        clinical_docs = [doc for doc in docs if doc.get('type') in ['paper', 'clinical_trial']]
        doc_content = "\n".join([f"- {doc['content'][:300]}" for doc in clinical_docs])
        
        prompt = self.prompts.clinical_analysis.format(
            query=query,
            context_understanding=context['analysis'],
            clinical_evidence=doc_content
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            return {"analysis": content, "evidence_count": len(clinical_docs)}
            
        except Exception as e:
            logger.warning(f"Clinical analysis failed: {e}")
            return {"analysis": "Clinical analysis unavailable", "evidence_count": 0}
    
    def _analyze_market(self, query: str, docs: List[Dict], context: Dict) -> Dict:
        """Step 3: Market Intelligence"""
        market_docs = [doc for doc in docs if doc.get('type') == 'market']
        doc_content = "\n".join([f"- {doc['content'][:300]}" for doc in market_docs])
        
        prompt = self.prompts.market_analysis.format(
            query=query,
            context_understanding=context['analysis'],
            market_intelligence=doc_content
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            return {"analysis": content, "market_docs": len(market_docs)}
            
        except Exception as e:
            logger.warning(f"Market analysis failed: {e}")
            return {"analysis": "Market analysis unavailable", "market_docs": 0}
    
    def _synthesize_decision(self, query: str, context: Dict, clinical: Dict, market: Dict) -> Dict:
        """Step 4: Decision Synthesis"""
        prompt = self.prompts.decision_synthesis.format(
            query=query,
            context_analysis=context['analysis'],
            clinical_analysis=clinical['analysis'],
            market_analysis=market['analysis']
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            return {"synthesis": content}
            
        except Exception as e:
            logger.warning(f"Decision synthesis failed: {e}")
            return {"synthesis": "Decision synthesis unavailable"}
    
    def _format_response(self, query: str, docs: List[Dict], context: Dict, 
                        clinical: Dict, market: Dict, decision: Dict,
                        from_knowledge_base: bool = True) -> Dict:
        """Format structured response"""
        
        # Extract key information (simplified parsing)
        clinical_text = clinical.get('analysis', '')
        market_text = market.get('analysis', '')
        decision_text = decision.get('synthesis', '')
        
        # Determine viability
        viability = "Medium"
        if any(word in clinical_text.lower() for word in ['promising', 'effective', 'successful']):
            viability = "High"
        elif any(word in clinical_text.lower() for word in ['failed', 'ineffective', 'toxic']):
            viability = "Low"
        
        # Extract risks
        risks = []
        if 'toxicity' in clinical_text.lower():
            risks.append('toxicity concerns')
        if 'side effect' in clinical_text.lower():
            risks.append('adverse effects')
        if 'trial' in clinical_text.lower() and 'fail' in clinical_text.lower():
            risks.append('trial failure history')
        if not risks:
            risks = ['standard development risks']
        
        # Market signal
        market_signal = "Moderate"
        if any(word in market_text.lower() for word in ['strong', 'growing', 'demand']):
            market_signal = "Strong"
        elif any(word in market_text.lower() for word in ['weak', 'declining', 'saturated']):
            market_signal = "Weak"
        
        # Recommendation
        recommendation = "Investigate Further"
        if viability == "High" and market_signal == "Strong":
            recommendation = "Proceed"
        elif viability == "Low" or market_signal == "Weak":
            recommendation = "Drop"
        
        # Confidence score - based on document relevance
        docs_used = context.get('docs_used', 0)
        evidence_count = clinical.get('evidence_count', 0)
        
        if docs_used == 0:
            confidence = 0.3  # Low confidence if no docs
        else:
            confidence = 0.5  # Base confidence
            if docs_used >= 3:
                confidence += 0.2
            if evidence_count >= 2:
                confidence += 0.2
            # Check average similarity score
            avg_similarity = sum(doc.get('similarity_score', 0) for doc in docs) / max(len(docs), 1)
            if avg_similarity > 0.5:
                confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        # Evidence files
        evidence_files = [doc.get('filename', f'doc_{i}') for i, doc in enumerate(docs[:3])]
        
        # Add confidence indicator to explanation
        confidence_label = "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.5 else "LOW"
        
        return {
            "clinical_viability": viability,
            "key_evidence": evidence_files,
            "major_risks": risks,
            "market_signal": market_signal,
            "recommendation": recommendation,
            "confidence_score": round(confidence, 2),
            "explanation": f"""
**Query Analysis**: {query}

**Clinical Assessment**: {clinical_text[:200]}...

**Market Intelligence**: {market_text[:200]}...

**Final Decision**: {decision_text[:200]}...

**Evidence Base**: {len(docs)} documents analyzed.
**Confidence**: {confidence_label} ({confidence*100:.0f}%) - {"Based on internal knowledge base" if from_knowledge_base else "Based on general AI knowledge"}
            """.strip()
        }
    
    def _fallback_response(self, error: str) -> Dict:
        """Fallback response for errors"""
        return {
            "clinical_viability": "Unknown",
            "key_evidence": [],
            "major_risks": ["analysis error"],
            "market_signal": "Unknown",
            "recommendation": "Manual Review Required",
            "confidence_score": 0.0,
            "explanation": f"Analysis failed: {error}. Please try again or contact support."
        }