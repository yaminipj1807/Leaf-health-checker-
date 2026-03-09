"""
Google Gemini AI Integration Module
Enhanced disease diagnosis, recommendations, and conversational AI support.
"""

import google.generativeai as genai
from pathlib import Path
import logging
import os
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Load API key from environment
GEMINI_API_KEY = os.getenv('GOOGLE_GEMINI_API_KEY', 'AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-pro')
TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.7'))
MAX_TOKENS = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '1000'))


class GeminiAIEngine:
    """Google Gemini AI integration for enhanced recommendations."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """
        Initialize Gemini AI engine.
        
        Args:
            api_key: Google Gemini API key
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self.chat_history = []
            logger.info(f"✓ Gemini AI initialized with model: {GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing Gemini AI: {str(e)}")
            self.model = None
    
    def generate_disease_explanation(self, 
                                    disease_name: str,
                                    plant_name: str,
                                    severity: str,
                                    affected_percentage: float) -> str:
        """
        Generate detailed disease explanation using Gemini AI.
        
        Args:
            disease_name: Name of detected disease
            plant_name: Plant species affected
            severity: Severity level (healthy, mild, moderate, severe, dying)
            affected_percentage: Percentage of affected area
            
        Returns:
            str: Detailed disease explanation
        """
        if self.model is None:
            return self._get_fallback_explanation(disease_name, severity)
        
        try:
            prompt = f"""
            As an expert plant pathologist, provide a detailed but concise explanation for:
            
            Disease: {disease_name}
            Plant: {plant_name}
            Severity Level: {severity}
            Affected Area: {affected_percentage:.1f}%
            
            Include:
            1. What causes this disease
            2. How it spreads
            3. Signs and symptoms (2-3 lines)
            4. Why it's at {severity} level
            5. Risk if left untreated
            
            Keep response under 150 words. Use simple, farmer-friendly language.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
            )
            
            explanation = response.text if response.text else self._get_fallback_explanation(disease_name, severity)
            logger.info(f"Generated disease explanation for {disease_name}")
            return explanation
        
        except Exception as e:
            logger.error(f"Error generating disease explanation: {str(e)}")
            return self._get_fallback_explanation(disease_name, severity)
    
    def generate_personalized_tips(self,
                                   disease_name: str,
                                   plant_name: str,
                                   severity: str,
                                   affected_percentage: float,
                                   default_tips: List[str]) -> Dict[str, any]:
        """
        Generate AI-powered personalized rescue tips.
        
        Args:
            disease_name: Disease name
            plant_name: Plant species
            severity: Severity level
            affected_percentage: Percentage affected
            default_tips: Default tips as fallback
            
        Returns:
            dict: Enhanced tips with priority and urgency
        """
        if self.model is None:
            return self._get_fallback_tips(disease_name, severity, default_tips)
        
        try:
            prompt = f"""
            Generate 5 prioritized rescue tips for a farmer facing this plant disease:
            
            Plant: {plant_name}
            Disease: {disease_name}
            Severity: {severity}
            Affected Area: {affected_percentage:.1f}%
            
            Format each tip as:
            - [PRIORITY: HIGH/MEDIUM/LOW] - [ACTION]: Description (max 1 line)
            
            Focus on:
            - Immediate actions for {severity} severity
            - Affordable and practical solutions
            - Prevention of disease spread
            - Recovery timeline expectations
            
            Default tips to consider:
            {chr(10).join(default_tips)}
            
            Keep ALL tips under 200 words total. Be specific about timing and dosages where applicable.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
            )
            
            enhanced_tips = response.text if response.text else None
            
            if enhanced_tips:
                logger.info(f"Generated enhanced tips for {disease_name}")
                return {
                    'enhanced_tips': enhanced_tips,
                    'default_tips': default_tips,
                    'status': 'success'
                }
            else:
                return self._get_fallback_tips(disease_name, severity, default_tips)
        
        except Exception as e:
            logger.error(f"Error generating personalized tips: {str(e)}")
            return self._get_fallback_tips(disease_name, severity, default_tips)
    
    def answer_plant_question(self, question: str, context: Optional[Dict] = None) -> str:
        """
        Answer user questions about plant health using Gemini AI.
        
        Args:
            question: User's question
            context: Optional context about diagnosed disease
            
        Returns:
            str: AI-generated answer
        """
        if self.model is None:
            return "AI Assistant temporarily unavailable. Please try again later."
        
        try:
            context_str = ""
            if context:
                context_str = f"""
                Recent diagnosis context:
                - Plant: {context.get('plant', 'N/A')}
                - Disease: {context.get('disease', 'N/A')}
                - Severity: {context.get('severity', 'N/A')}
                """
            
            prompt = f"""
            You are an expert agricultural advisor. Answer this farmer's question about plant health.
            
            {context_str}
            
            Question: {question}
            
            Provide:
            1. Clear, practical answer
            2. Step-by-step guidance if applicable
            3. Safety precautions if needed
            4. When to seek professional help
            
            Keep response under 200 words. Use simple language for non-experts.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
            )
            
            answer = response.text if response.text else "Unable to generate answer. Please try again."
            logger.info(f"Answered question: {question[:50]}...")
            return answer
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error processing your question: {str(e)}"
    
    def chat(self, message: str) -> str:
        """
        Multi-turn conversation with Gemini AI.
        
        Args:
            message: User message
            
        Returns:
            str: AI response
        """
        if self.model is None:
            return "AI Assistant unavailable. Please check API configuration."
        
        try:
            # Start conversation if first message
            if not self.chat_history:
                chat = self.model.start_chat(history=[])
            else:
                chat = self.model.start_chat(history=self.chat_history)
            
            system_instruction = """
            You are an expert agricultural advisor specializing in plant disease diagnosis and treatment.
            Be friendly, practical, and provide actionable advice for farmers and gardeners.
            Ask clarifying questions if needed. Suggest professional consultation for severe cases.
            """
            
            response = chat.send_message(
                f"{system_instruction}\n\nUser: {message}",
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
            )
            
            assistant_message = response.text if response.text else "No response generated."
            
            # Store conversation history
            self.chat_history.append({
                'role': 'user',
                'parts': message
            })
            self.chat_history.append({
                'role': 'model',
                'parts': assistant_message
            })
            
            logger.info(f"Chat message processed")
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_care_plan(self,
                          plant_name: str,
                          disease_name: str,
                          severity: str) -> str:
        """
        Generate comprehensive plant care plan.
        
        Args:
            plant_name: Plant species
            disease_name: Disease name
            severity: Severity level
            
        Returns:
            str: Detailed care plan
        """
        if self.model is None:
            return "Unable to generate care plan. Please try again."
        
        try:
            prompt = f"""
            Create a detailed 7-day care plan for recovering a {plant_name} with {disease_name} ({severity}):
            
            Include:
            1. Daily actions (watering, spraying, pruning)
            2. Environmental conditions to maintain
            3. Products to use with dosages
            4. Monitoring checkpoints
            5. When to escalate to professional
            
            Format as a structured day-by-day plan.
            Keep under 300 words.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=1500
                )
            )
            
            plan = response.text if response.text else "Unable to generate plan."
            logger.info(f"Generated care plan for {plant_name}")
            return plan
        
        except Exception as e:
            logger.error(f"Error generating care plan: {str(e)}")
            return "Error generating care plan. Please try again."
    
    def identify_preventive_measures(self,
                                     plant_name: str,
                                     disease_name: str,
                                     climate_zone: str = "Tropical") -> List[str]:
        """
        Get preventive measures for disease.
        
        Args:
            plant_name: Plant species
            disease_name: Disease to prevent
            climate_zone: Geographic/climate zone
            
        Returns:
            list: Preventive measures
        """
        if self.model is None:
            return ["Maintain proper plant spacing", "Ensure good air circulation", "Regular monitoring"]
        
        try:
            prompt = f"""
            List 5 specific preventive measures to avoid {disease_name} in {plant_name} grown in {climate_zone} climate.
            
            Format as bullet points, one per line.
            Be specific and practical.
            Include seasonal timing where applicable.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=500
                )
            )
            
            if response.text:
                measures = [line.strip() for line in response.text.split('\n') if line.strip() and line.strip().startswith('•')]
                logger.info(f"Generated preventive measures for {disease_name}")
                return measures if measures else ["Maintain plant health", "Regular monitoring", "Proper hygiene"]
            else:
                return ["Unable to generate measures", "Please try again"]
        
        except Exception as e:
            logger.error(f"Error identifying preventive measures: {str(e)}")
            return ["Consult with agricultural expert"]
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    @staticmethod
    def _get_fallback_explanation(disease_name: str, severity: str) -> str:
        """Fallback disease explanation."""
        explanations = {
            'late_blight': "Late Blight is a severe fungal disease caused by Phytophthora infestans. It spreads rapidly in cool, wet conditions and can destroy entire plants if untreated. Infected leaves show water-soaked spots that turn brown and develop a white fungal layer on the underside. The disease spreads through water splash and airborne spores.",
            'early_blight': "Early Blight causes circular brown spots with concentric rings on older leaves. It's less aggressive than late blight but still requires treatment. The disease overwinters in plant debris and spreads during warm, humid periods.",
            'powdery_mildew': "Powdery Mildew appears as white powder on leaf surfaces. It thrives in warm, dry conditions with high humidity. While rarely fatal, it weakens the plant and reduces fruit quality if untreated.",
            'septoria_leaf_spot': "Septoria Leaf Spot creates small circular lesions with gray centers and dark borders. It primarily affects older foliage and spreads through water splash and contact.",
            'rust': "Rust appears as rusty-orange pustules on leaf undersides. It's a fungal disease that weakens the plant by reducing photosynthetic capacity."
        }
        
        base_explanation = explanations.get(disease_name, f"{disease_name} is a plant disease requiring treatment.")
        
        severity_notes = {
            'healthy': " The plant shows no symptoms.",
            'mild': " Early intervention can quickly resolve this.",
            'moderate': " Prompt action is needed to prevent progression.",
            'severe': " Intensive treatment is required immediately.",
            'dying': " This requires emergency action to save the plant."
        }
        
        return base_explanation + severity_notes.get(severity, "")
    
    @staticmethod
    def _get_fallback_tips(disease_name: str, severity: str, default_tips: List[str]) -> Dict:
        """Fallback tips."""
        return {
            'enhanced_tips': "\n".join([f"- {tip}" for tip in default_tips]),
            'default_tips': default_tips,
            'status': 'fallback'
        }


# Initialize global Gemini instance
try:
    gemini_engine = GeminiAIEngine()
except Exception as e:
    logger.warning(f"Gemini AI initialization warning: {str(e)}")
    gemini_engine = None


def get_gemini_engine() -> Optional[GeminiAIEngine]:
    """Get global Gemini AI engine instance."""
    global gemini_engine
    if gemini_engine is None:
        try:
            gemini_engine = GeminiAIEngine()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini engine: {str(e)}")
    return gemini_engine
