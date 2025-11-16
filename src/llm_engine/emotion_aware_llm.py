"""
Emotion-Aware LLM Engine.
Integrates language models with emotional context for adaptive responses.
"""

import torch
from typing import Dict, Optional, List
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig
)

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class EmotionAwareLLM:
    """
    LLM with emotion-aware response generation.
    """

    # System prompt template
    SYSTEM_PROMPT = """You are an empathetic AI companion designed to provide supportive, personalized interactions.
You can sense the user's emotional state through their facial expressions, voice tone, and messages.

Your role is to:
1. Acknowledge and validate the user's emotions
2. Adapt your communication style based on their emotional state
3. Provide helpful, contextual responses
4. Maintain a caring and supportive presence

Always be respectful, genuine, and helpful."""

    # Response mode prompts
    MODE_PROMPTS = {
        'supportive': """Adopt an empathetic and caring tone. Provide emotional support and validation.
Be understanding and gentle. Offer comfort and encouragement.""",

        'cheerful': """Adopt an upbeat and motivating tone. Be positive and energizing.
Share enthusiasm and encouragement. Help lift the user's spirits.""",

        'calm': """Adopt a soothing and grounding tone. Be peaceful and reassuring.
Help the user feel centered and relaxed. Speak in a measured, calming manner.""",

        'concise': """Adopt a brief and efficient tone. Be direct and to-the-point.
Provide clear, actionable information without unnecessary elaboration.""",

        'neutral': """Adopt a professional assistant tone. Be helpful and informative.
Provide balanced, objective responses."""
    }

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the emotion-aware LLM.

        Args:
            model_name: Model name or path (uses config if None)
        """
        self.config = get_config()
        self.llm_config = self.config.get_section("llm")

        # Model configuration
        if model_name is None:
            model_name = self.llm_config.get(
                "model_name",
                "microsoft/Phi-3-mini-4k-instruct"
            )

        self.model_name = model_name
        self.device = self._get_device()

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self._load_model()

        # Conversation history
        self.conversation_history: List[Dict] = []
        self.max_history = 10

        logger.info(f"Emotion-aware LLM initialized with model: {model_name}")

    def _get_device(self) -> str:
        """Determine compute device."""
        device = self.llm_config.get("device", "auto")

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load the LLM and tokenizer."""
        try:
            logger.info(f"Loading LLM: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            # Generation configuration
            self.generation_config = GenerationConfig(
                max_length=self.llm_config.get("max_length", 512),
                temperature=self.llm_config.get("temperature", 0.7),
                top_p=self.llm_config.get("top_p", 0.9),
                top_k=self.llm_config.get("top_k", 50),
                repetition_penalty=self.llm_config.get("repetition_penalty", 1.1),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def generate_response(
        self,
        user_message: str,
        emotion_context: Optional[Dict] = None,
        response_mode: Optional[str] = None,
        include_history: bool = True
    ) -> str:
        """
        Generate an emotion-aware response.

        Args:
            user_message: User's message
            emotion_context: Current emotion data
            response_mode: Response mode (supportive, cheerful, calm, etc.)
            include_history: Whether to include conversation history

        Returns:
            Generated response text
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(
                user_message,
                emotion_context,
                response_mode,
                include_history
            )

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract just the assistant's response
            response = self._extract_response(response, prompt)

            # Add to conversation history
            self._add_to_history(user_message, response, emotion_context)

            logger.debug(f"Generated response (mode: {response_mode})")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

    def _build_prompt(
        self,
        user_message: str,
        emotion_context: Optional[Dict],
        response_mode: Optional[str],
        include_history: bool
    ) -> str:
        """
        Build the complete prompt with emotion context.

        Args:
            user_message: User's message
            emotion_context: Emotion data
            response_mode: Response mode
            include_history: Include conversation history

        Returns:
            Formatted prompt
        """
        # Start with system prompt
        prompt_parts = [self.SYSTEM_PROMPT]

        # Add emotion context if available
        if emotion_context:
            emotion_prompt = self._format_emotion_context(emotion_context)
            prompt_parts.append(emotion_prompt)

        # Add response mode instruction
        if response_mode and response_mode in self.MODE_PROMPTS:
            prompt_parts.append(self.MODE_PROMPTS[response_mode])

        # Add conversation history
        if include_history and self.conversation_history:
            history_text = self._format_history()
            prompt_parts.append("\n\nConversation History:")
            prompt_parts.append(history_text)

        # Add current user message
        prompt_parts.append(f"\n\nUser: {user_message}\n\nAssistant:")

        return "\n".join(prompt_parts)

    def _format_emotion_context(self, emotion_context: Dict) -> str:
        """
        Format emotion context for the prompt.

        Args:
            emotion_context: Emotion data

        Returns:
            Formatted emotion context string
        """
        parts = ["\n\nCurrent User Emotional State:"]

        # Dominant emotion
        dominant = emotion_context.get('dominant_emotion', 'unknown')
        confidence = emotion_context.get('confidence', 0.0)
        parts.append(f"- Primary emotion: {dominant} (confidence: {confidence:.1%})")

        # Emotion distribution
        emotions = emotion_context.get('emotions', {})
        if emotions:
            top_emotions = sorted(
                emotions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            emotions_str = ", ".join([f"{e}: {s:.1%}" for e, s in top_emotions])
            parts.append(f"- Emotion signals: {emotions_str}")

        # Modalities used
        modalities = emotion_context.get('modalities_used', [])
        if modalities:
            parts.append(f"- Detected from: {', '.join(modalities)}")

        # Additional context
        if 'stability' in emotion_context:
            stability = emotion_context['stability']
            parts.append(f"- Emotional stability: {stability:.1%}")

        return "\n".join(parts)

    def _format_history(self, max_turns: int = 5) -> str:
        """
        Format conversation history.

        Args:
            max_turns: Maximum conversation turns to include

        Returns:
            Formatted history string
        """
        recent_history = self.conversation_history[-max_turns:]

        history_lines = []
        for turn in recent_history:
            history_lines.append(f"User: {turn['user_message']}")
            history_lines.append(f"Assistant: {turn['assistant_response']}")

        return "\n".join(history_lines)

    def _extract_response(self, full_output: str, prompt: str) -> str:
        """
        Extract assistant response from full model output.

        Args:
            full_output: Complete model output
            prompt: Original prompt

        Returns:
            Extracted response
        """
        # Remove the prompt
        if full_output.startswith(prompt):
            response = full_output[len(prompt):].strip()
        else:
            # Try to find "Assistant:" marker
            parts = full_output.split("Assistant:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_output.strip()

        # Clean up the response
        response = response.strip()

        # Remove any trailing user prompts
        if "\nUser:" in response:
            response = response.split("\nUser:")[0].strip()

        return response

    def _add_to_history(
        self,
        user_message: str,
        assistant_response: str,
        emotion_context: Optional[Dict]
    ):
        """
        Add conversation turn to history.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            emotion_context: Emotion context
        """
        self.conversation_history.append({
            'user_message': user_message,
            'assistant_response': assistant_response,
            'emotion_context': emotion_context,
            'timestamp': datetime.now()
        })

        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def adjust_generation_params(
        self,
        response_mode: str
    ):
        """
        Adjust generation parameters based on response mode.

        Args:
            response_mode: Response mode
        """
        mode_config = self.llm_config.get('modes', {}).get(response_mode, {})

        if 'temperature' in mode_config:
            self.generation_config.temperature = mode_config['temperature']

        logger.debug(
            f"Adjusted generation params for mode: {response_mode}, "
            f"temp: {self.generation_config.temperature}"
        )

    def get_conversation_history(self) -> List[Dict]:
        """
        Get the conversation history.

        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("LLM resources cleaned up")


def determine_response_mode(emotion_data: Optional[Dict]) -> str:
    """
    Determine appropriate response mode based on emotion.

    Args:
        emotion_data: Current emotion data

    Returns:
        Response mode name
    """
    if emotion_data is None:
        return 'neutral'

    # Get emotion-to-mode mapping from config
    config = get_config()
    emotion_mode_map = config.get(
        'adaptive_response.emotion_mode_map',
        {
            'sad': 'supportive',
            'angry': 'calm',
            'fearful': 'calm',
            'happy': 'cheerful',
            'surprised': 'cheerful',
            'neutral': 'neutral',
            'disgusted': 'supportive'
        }
    )

    dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')

    return emotion_mode_map.get(dominant_emotion, 'neutral')
