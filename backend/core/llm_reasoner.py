"""
LLM Reasoner Module for AI Policy Query System

This module handles the reasoning and response generation using LLMs to provide
structured answers to insurance policy questions based on retrieved context.
"""

import json
import re
import os
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import openai


@dataclass
class PolicyResponse:
    """
    Structured response for policy queries.
    
    Attributes:
        decision: The main answer to the user's question
        justification: Detailed explanation based on policy text
        reference: Specific section or clause reference
        confidence: Confidence score (0-1)
        additional_info: Optional additional relevant information
    """
    decision: str
    justification: str
    reference: str
    confidence: float = 0.0
    additional_info: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for JSON serialization."""
        result = {
            "decision": self.decision,
            "justification": self.justification,
            "reference": self.reference
        }
        if self.confidence > 0:
            result["confidence"] = self.confidence
        if self.additional_info:
            result["additional_info"] = self.additional_info
        return result


class PolicyReasoner:
    """
    Handles reasoning about insurance policy questions using LLMs.
    
    This class provides structured analysis of policy documents to answer
    user questions with proper justification and references.
    """
    
    def __init__(
        self,
        use_perplexity: bool = True,
        perplexity_api_key: Optional[str] = None,
        model_name: str = "sonar-pro",
        temperature: float = 0.1
    ):
        """
        Initialize the policy reasoner.
        
        Args:
            use_perplexity: Whether to use Perplexity models
            perplexity_api_key: Perplexity API key
            model_name: Name of the Perplexity model to use
            temperature: Temperature for response generation
        """
        self.use_perplexity = use_perplexity
        self.model_name = model_name
        self.temperature = temperature
        
        if use_perplexity:
            if perplexity_api_key:
                self.perplexity_api_key = perplexity_api_key
            elif os.getenv("PERPLEXITY_API_KEY"):
                self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
            elif os.getenv("API_KEY"):
                self.perplexity_api_key = os.getenv("API_KEY")
            else:
                raise ValueError("Perplexity API key required when use_perplexity=True. Set PERPLEXITY_API_KEY or API_KEY environment variable.")
        else:
            # For local models, we'll use a simple implementation
            # In production, you might want to use Hugging Face transformers
            print("Note: Local LLM support requires additional setup")
    
    def analyze_query(self, query: str, context: str, document_name: str = "Policy Document") -> PolicyResponse:
        """
        Analyze a policy query using retrieved context.
        
        Args:
            query: User's question about the policy
            context: Retrieved relevant text from the policy
            document_name: Name of the source document
            
        Returns:
            PolicyResponse with structured analysis
        """
        # Create the analysis prompt
        prompt = self._create_analysis_prompt(query, context, document_name)
        
        # Generate response using LLM
        if self.use_perplexity:
            raw_response = self._generate_perplexity_response(prompt)
        else:
            raw_response = self._generate_local_response(prompt)
        
        # Parse and validate the response
        parsed_response = self._parse_response(raw_response, query)
        
        return parsed_response
    
    def _create_analysis_prompt(self, query: str, context: str, document_name: str) -> str:
        """
        Create a structured prompt for policy analysis.
        
        Args:
            query: User's question
            context: Retrieved policy context
            document_name: Source document name
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert insurance policy analyst. Your task is to analyze insurance policy documents and provide accurate, helpful answers to user questions.

**INSTRUCTIONS:**
1. Carefully read the provided policy context
2. Answer the user's question based ONLY on the information provided
3. If the answer is not clear from the context, state that explicitly
4. Provide specific references to sections when possible
5. Respond in the exact JSON format shown below

**USER QUESTION:**
{query}

**POLICY CONTEXT FROM {document_name}:**
{context}

**REQUIRED RESPONSE FORMAT:**
Return ONLY a valid JSON object with these exact fields:
{{
  "decision": "Clear, direct answer to the user's question",
  "justification": "Detailed explanation based on the policy text, including specific quotes when relevant",
  "reference": "Specific section, clause, or page reference from the policy"
}}

**IMPORTANT GUIDELINES:**
- Be precise and factual - avoid speculation
- If information is incomplete, mention what additional details might be needed
- For coverage questions, clearly state whether something IS or IS NOT covered
- For monetary amounts, include specific figures when available
- For time-based questions, include exact waiting periods, terms, or deadlines
- If multiple sections are relevant, reference the most important one

**JSON RESPONSE:**"""

        return prompt
    
    def _generate_openai_response(self, prompt: str) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: The analysis prompt
            
        Returns:
            Raw response text
        """
        try:
            import time
            start_time = time.time()
            timeout = 30  # 30 second timeout
            
            # Use the new OpenAI API format for v1.0.0+
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise insurance policy analyst. Always respond with valid JSON in the requested format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1000,
                timeout=timeout
            )
            
            if time.time() - start_time > timeout:
                raise RuntimeError("OpenAI API call timed out")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error generating OpenAI response: {str(e)}")
    
    def _generate_perplexity_response(self, prompt: str) -> str:
        """
        Generate response using Perplexity API.
        
        Args:
            prompt: The analysis prompt
            
        Returns:
            Raw response text
        """
        try:
            import time
            start_time = time.time()
            timeout = 60  # 60 second timeout for Perplexity API
            
            # Perplexity API endpoint
            url = "https://api.perplexity.ai/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise insurance policy analyst. Always respond with valid JSON in the requested format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": 2000,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if time.time() - start_time > timeout:
                raise RuntimeError("Perplexity API call timed out")
            
            if response.status_code == 401:
                raise RuntimeError(f"Perplexity API authentication failed. Please check your API key. Status: {response.status_code}")
            elif response.status_code != 200:
                raise RuntimeError(f"Perplexity API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise RuntimeError("Perplexity API returned empty response")
                
            return response_data["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            raise RuntimeError(f"Error generating Perplexity response: {str(e)}")
    
    def _generate_local_response(self, prompt: str) -> str:
        """
        Generate response using local LLM (placeholder implementation).
        
        Args:
            prompt: The analysis prompt
            
        Returns:
            Raw response text
        """
        # This is a placeholder for local LLM implementation
        # In production, you would integrate with Hugging Face transformers
        # or other local LLM solutions
        
        return '''
        {
          "decision": "Local LLM support not fully implemented",
          "justification": "This is a placeholder response. To use local LLMs, integrate with Hugging Face transformers or similar libraries.",
          "reference": "Implementation Note"
        }
        '''
    
    def _parse_response(self, raw_response: str, original_query: str) -> PolicyResponse:
        """
        Parse and validate the LLM response.
        
        Args:
            raw_response: Raw response from LLM
            original_query: Original user query for context
            
        Returns:
            Parsed PolicyResponse object
        """
        try:
            # Clean the response - remove any markdown formatting
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = cleaned_response
            
            # Parse JSON
            response_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["decision", "justification", "reference"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create PolicyResponse object
            policy_response = PolicyResponse(
                decision=response_data["decision"],
                justification=response_data["justification"],
                reference=response_data["reference"],
                confidence=response_data.get("confidence", 0.8),  # Default confidence
                additional_info=response_data.get("additional_info")
            )
            
            return policy_response
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback response if parsing fails
            return PolicyResponse(
                decision="Unable to process the query properly",
                justification=f"There was an error processing your question: {str(e)}. The system may need more specific policy context to provide an accurate answer.",
                reference="System Error",
                confidence=0.0
            )
    
    def batch_analyze(self, queries_and_contexts: List[tuple]) -> List[PolicyResponse]:
        """
        Analyze multiple queries in batch.
        
        Args:
            queries_and_contexts: List of (query, context, document_name) tuples
            
        Returns:
            List of PolicyResponse objects
        """
        responses = []
        
        for query, context, doc_name in queries_and_contexts:
            try:
                response = self.analyze_query(query, context, doc_name)
                responses.append(response)
            except Exception as e:
                # Add error response for failed queries
                error_response = PolicyResponse(
                    decision="Analysis failed",
                    justification=f"Could not analyze query due to error: {str(e)}",
                    reference="System Error",
                    confidence=0.0
                )
                responses.append(error_response)
        
        return responses
    
    def evaluate_confidence(self, query: str, context: str, response: PolicyResponse) -> float:
        """
        Evaluate confidence in the response based on context quality.
        
        Args:
            query: Original query
            context: Retrieved context
            response: Generated response
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Context length and quality
        if len(context) > 500:
            confidence_factors.append(0.3)
        elif len(context) > 200:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Factor 2: Specific references in response
        if any(keyword in response.reference.lower() for keyword in ['section', 'clause', 'page']):
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)
        
        # Factor 3: Query-context relevance
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        overlap = len(query_words.intersection(context_words))
        if overlap > 3:
            confidence_factors.append(0.4)
        elif overlap > 1:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)
        
        return min(1.0, sum(confidence_factors))


def analyze_policy_question(
    query: str,
    context: str,
    document_name: str = "Policy Document",
    use_openai: bool = True,
    model_name: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Convenience function to analyze a single policy question.
    
    Args:
        query: User's question
        context: Retrieved policy context
        document_name: Source document name
        use_openai: Whether to use OpenAI
        model_name: Model name to use
        
    Returns:
        Dictionary with structured response
    """
    reasoner = PolicyReasoner(
        use_openai=use_openai,
        model_name=model_name
    )
    
    response = reasoner.analyze_query(query, context, document_name)
    return response.to_dict()


# Example usage and testing
if __name__ == "__main__":
    # Sample policy context for testing
    sample_context = """
    Section 4.2 - Maternity Coverage: 
    Maternity expenses including prenatal care, delivery, and postnatal care are covered under this policy. 
    Coverage begins after a mandatory 2-year waiting period from the policy effective date. 
    Maximum benefit amount is $10,000 per pregnancy. This includes hospital charges, doctor fees, and necessary medical procedures.
    
    Section 4.3 - Exclusions:
    The following maternity-related expenses are not covered: elective cesarean sections without medical necessity, 
    fertility treatments, and complications arising from high-risk pregnancies not disclosed during enrollment.
    """
    
    # Test queries
    test_queries = [
        "Is maternity covered in this policy?",
        "What is the waiting period for maternity benefits?",
        "How much coverage is provided for pregnancy?",
        "Are fertility treatments covered?"
    ]
    
    try:
        # Note: This will only work if you have OpenAI API key set
        print("Testing Policy Reasoner...")
        
        for query in test_queries:
            print(f"\n🔍 Question: {query}")
            
            try:
                result = analyze_policy_question(
                    query=query,
                    context=sample_context,
                    document_name="Sample Health Policy",
                    use_openai=True  # Set to False to test local mode
                )
                
                print(f"✅ Decision: {result['decision']}")
                print(f"📋 Justification: {result['justification'][:100]}...")
                print(f"📖 Reference: {result['reference']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"Setup error: {e}")
        print("Note: Ensure OpenAI API key is configured to test with real LLM") 