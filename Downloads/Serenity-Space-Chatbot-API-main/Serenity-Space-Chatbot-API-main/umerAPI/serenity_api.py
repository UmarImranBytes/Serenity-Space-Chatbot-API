import random
import logging
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
import os
import json

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# Initialize FastAPI app
app = FastAPI(title="SerenityBot API", description="API for emotional wellness support", version="1.0.0")

# Pydantic model for request body
class UserInput(BaseModel):
    mood: str
    age: Optional[int] = None
    reason: Optional[str] = None
    user_id: Optional[str] = None
    input_text: Optional[str] = None  # For follow-up inputs like 'mindfulness'
    conversation_id: Optional[str] = None
    stop: Optional[bool] = False

class SerenitySupport:
    def __init__(self):
        """Initialize the AI model with dynamic and context-aware emotional wellness features."""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.85,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 2000,
                    "response_mime_type": "text/plain",
                },
                system_instruction=(
                    "You are SerenityBot, a compassionate, conversational AI assistant for emotional wellness. "
                    "Analyze user input to detect mood, tone, age, and reason accurately. Deliver empathetic, actionable responses tailored to the user's emotional state. "
                    "For negative moods (e.g., sad, anxious), use a warm, validating tone with phrases like 'I'm really sorry you're going through this' and 'You're not alone.' "
                    "For positive moods (e.g., happy), use an uplifting, celebratory tone with phrases like 'I’m so thrilled to hear you’re feeling happy!' and 'Let’s keep that joy shining!' "
                    "Structure responses with a heartfelt introduction, followed by 5-6 numbered steps with emojis (e.g., 🌿, 🧠 for negative moods; 🌞, 🎉 for positive moods) that guide the user through immediate and long-term mental health strategies. "
                    "Tailor responses to the user's mood, age, and reason (e.g., relationship issues or successes). Offer diverse follow-up options. "
                    "Use conversation history for continuity, avoid repetitive responses, and ask engaging, context-specific follow-up questions."
                )
            )
            self.user_profiles: Dict[str, Dict] = {}
            self.conversation_history: Dict[str, List[Dict]] = {}
            self.conversation_state: Dict[str, Dict] = {}
            
            # Move static dictionaries to __init__ for efficiency
            self.mood_keywords = {
                "anxious": ["anxious", "nervous", "worried", "overwhelmed", "panicked", "tense", "afraid"],
                "sad": ["sad", "down", "depressed", "low", "unhappy", "lonely", "heartbroken"],
                "happy": ["happy", "joyful", "excited", "content", "great", "awesome", "thrilled"],
                "angry": ["angry", "mad", "irritated", "furious", "annoyed", "upset", "enraged"],
                "stressed": ["stressed", "pressure", "overloaded", "busy", "deadline", "strained"],
                "curious": ["curious", "wonder", "interested", "intrigued", "fascinated", "questioning"],
                "frustrated": ["frustrated", "annoyed", "stuck", "irritating", "blocked", "exasperated"],
                "betrayed": ["cheating", "betrayed", "disloyal", "trust", "hurt", "backstabbed"]
            }

            self.tone_keywords = {
                "positive": ["great", "awesome", "happy", "excited", "wonderful", "fantastic"],
                "negative": ["bad", "terrible", "sad", "angry", "stressed", "awful"],
                "neutral": ["okay", "fine", "not sure", "just", "alright"],
                "urgent": ["urgent", "help", "need", "immediately", "now"],
                "reflective": ["think", "wonder", "reflect", "feel", "consider"]
            }

            self.context_keywords = {
                "work_stress": ["work", "job", "deadline", "boss", "project"],
                "exam_stress": ["exam", "study", "test", "school", "grade"],
                "emotional_support": ["emotional", "feel", "heart", "support", "hurt"],
                "social": ["friend", "family", "relationship", "partner", "social"],
                "personal_growth": ["grow", "learn", "improve", "goal", "change"]
            }

            self.special_inputs = {
                "mindfulness": {
                    "response": ["Let’s try a guided mindfulness exercise. Close your eyes if comfortable, and focus on your breath for 1 minute. Inhale for 4, exhale for 6. Notice thoughts but let them pass like clouds. Gently return to your breath if your mind wanders."],
                    "follow_up": "How did that feel? Want another mindfulness exercise or explore a different tool? (Type 'mindfulness', 'tools', or share more.)"
                },
                "breathe": {
                    "response": ["Let's do a simple breathing exercise together. Breathe in deeply through your nose for 4 seconds... hold for 4 seconds... and slowly exhale through your mouth for 6 seconds. Repeat this 3 times."],
                    "follow_up": "Do you feel a bit more relaxed? We can try another round, or explore something else. (Type 'breathe', 'explore', or share more.)"
                },
                "cbt": {
                    "response": ["CBT (Cognitive Behavioral Therapy) techniques can be very helpful. Try catching a negative thought you had today. Ask yourself: 'Is there actual evidence for this thought, or am I assuming the worst?' Try to write down a more balanced perspective."],
                    "follow_up": "Would you like to try another CBT exercise or dive deeper into one specific thought? (Type 'cbt', 'deep dive', or share more.)"
                },
                "tools": {
                    "response": ["Let’s explore long-term coping tools. Try journaling daily: Write 3 things you’re grateful for and 1 thing you’re proud of. Or use a CBT technique—when a negative thought arises, ask, 'Is this 100% true?' and reframe it. Apps like Headspace or Calm can also help."],
                    "follow_up": "Want to dive into journaling, try a CBT exercise, or check out a mental health app? (Type 'journal', 'cbt', 'app', or your thoughts.)"
                },
                "plan": {
                    "response": ["Let’s create a plan! Pick 3 activities that spark joy or bring peace—like a walk, music, or calling a friend. Schedule one for today and one for tomorrow. Add a daily affirmation."],
                    "follow_up": "How’s the plan feeling? Want to tweak it or add a new activity? (Type 'tweak', 'add', or share more.)"
                },
                "resources": {
                    "response": ["Here are some support resources: Text 'HOME' to 741741 (Crisis Text Line) for free 24/7 support in the US. Or explore therapy options at BetterHelp or Talkspace. For local counselors, check Psychology Today’s directory."],
                    "follow_up": "Would you like more resource details or help finding local support? (Type 'details', 'local', or your thoughts.)"
                },
                "talk": {
                    "response": ["I’m here—what’s on your mind or heart right now? Share as much or as little as feels right, and I’ll listen."],
                    "follow_up": "Thank you for sharing. Want to keep talking, try a grounding exercise, or explore an idea? (Type 'talk', 'exercise', 'idea', or your thoughts.)"
                },
                "connect": {
                    "response": ["Connecting with others can make a huge difference! Reach out to a friend who gets you, join a fun online community like a hobby group, or plan a hangout. Small connections keep the good vibes going."],
                    "follow_up": "Want ideas for finding a community or tips for connecting? (Type 'community', 'reach out', or share more.)"
                },
                "goal": {
                    "response": ["Let’s set a goal. Try a daily practice: Note 3 things that make you smile or feel calm. Or plan to try one new activity this week, like a hobby or outing."],
                    "follow_up": "How does that goal feel? Want to set another or adjust this one? (Type 'another', 'adjust', or your thoughts.)"
                },
                "gratitude": {
                    "response": ["Let’s deepen your focus with gratitude! Write or think of 3 things you’re thankful for today—maybe a person, a moment, or something small. Feel how it warms your heart."],
                    "follow_up": "How did that feel? Want to try another gratitude exercise or plan an activity? (Type 'gratitude', 'plan', or share more.)"
                },
                "share": {
                    "response": ["I’d love to hear more! What’s the best part of this moment or your situation right now? Share as much as you’d like."],
                    "follow_up": "Thanks for sharing! Want to keep talking, or plan something fun? (Type 'talk', 'plan', or your thoughts.)"
                },
                "creative": {
                    "response": ["Let’s channel your energy creatively! Try journaling about this moment, sketching something that represents your mood, or making a playlist of songs that resonate with you."],
                    "follow_up": "How was that? Want another creative idea or to share your creation? (Type 'creative', 'share', or your thoughts.)"
                }
            }

            self.mood_library: Dict[str, Dict[str, List[str]]] = {
                "sad": {
                    "intros": [
                        "I'm really sorry you're feeling sad right now, especially because of {reason}. Relationships can be so hard, especially when you’re navigating them as a young adult, and it’s completely okay to feel this way. You’re not alone, and your heart is stronger than you might feel in this moment.",
                        "Feeling sad about {reason} can weigh so heavily, and I’m here to hold space for you. It’s okay if this feels overwhelming—your feelings are valid, and just reaching out here is a brave step."
                    ],
                    "steps": [
                        "🌿 Ground Your Body: Find a quiet spot and take 5 slow, deep breaths—inhale for 4, exhale for 6. Notice any tension (maybe in your chest or shoulders) and let it soften. If you haven’t eaten or hydrated, grab a small snack or water. Caring for your body can ease your mind a little.",
                        "🧠 Name Your Emotions: Ask yourself, 'What am I feeling right now?' It might be sadness, loneliness, or even anger. Try saying, 'I’m feeling sad because of {reason}.' If it’s hard to pin down, that’s okay—just being curious helps. Naming emotions can make them feel less overwhelming.",
                        "✍️ Release the Weight: Write down what’s on your heart, even if it’s just a few words in your phone (e.g., 'hurt,' 'lost'). If you’re up for it, talk to a trusted friend or family member. If not, try drawing or listening to music that matches your mood to let it out safely.",
                        "💡 Anchor in the Present: Try this grounding trick: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This helps calm racing thoughts and reminds you you’re here, in this moment, with control over small things.",
                        "🫂 Nurture Yourself: Choose one kind act for yourself. If you’re low-energy, wrap up in a blanket or watch a comforting show. If you feel up for it, take a short walk or text a friend who lifts you up. You deserve care, no matter how small it feels.",
                        "🌟 Build Toward Tomorrow: Think of one tiny step for tomorrow—like journaling for 5 minutes about what you want in life or repeating to yourself, 'I’m enough.' Sadness doesn’t define you, and you’re planting seeds for brighter days."
                    ],
                    "relationship_steps": [
                        "🌿 Soothe Your Body: Sit somewhere cozy and take 5 deep breaths, letting your shoulders drop. Relationship pain can feel like a physical ache, so grab a warm drink, water, or a snack to ground yourself. If you’re tense, try stretching gently for a minute.",
                        "🧠 Acknowledge the Hurt: Ask, 'What’s the hardest part of this relationship issue?' Maybe it’s feeling unvalued or betrayed. Say to yourself, 'I’m sad because my relationship hurts right now.' Naming the pain can make it feel less like it’s controlling you.",
                        "✍️ Express Safely: Write a letter to yourself about what you deserve in a relationship—love, respect, honesty. Don’t send it; this is for your healing. Or jot down a few words about how you feel (e.g., 'broken,' 'hopeful'). If writing’s not your thing, try talking to a trusted friend or even singing a song that resonates.",
                        "💡 Stay in the Now: Ground yourself with this: Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste. This can ease the spiral of ‘what-ifs’ about your relationship and remind you you’re still you, right here.",
                        "🫂 Care for Your Heart: Do something kind for yourself—watch a favorite movie, take a warm shower, or text a friend who makes you laugh. If you’re up for it, try a short walk to clear your head. You’re worthy of love, starting with your own.",
                        "🌟 Look Forward: Reflect on one small boundary you could set to protect your heart (e.g., 'I’ll say no to things that don’t feel right'). Or repeat to yourself, 'I’m enough, just as I am.' You’re building a stronger you, one step at a time."
                    ],
                    "follow_ups": [
                        "Would you like to try a guided mindfulness exercise or explore long-term tools like journaling or CBT techniques? (Type 'mindfulness', 'tools', or share more.)",
                        "I’m here if you want to create a self-care plan or find support resources, like a therapist or a community group. What feels right? (Type 'plan', 'resources', or your thoughts.)",
                        "Want to talk more about your relationship or connect with others who get it, like a support group? (Type 'talk', 'connect', or share more.)",
                        "How about setting a small goal for your emotional wellness, like practicing self-compassion daily? Or we can explore something else. (Type 'goal', 'explore', or your thoughts.)"
                    ]
                },
                "happy": {
                    "intros": [
                        "I’m so thrilled to hear you’re feeling happy right now, especially because of {reason}! A joyful moment in a relationship is something to cherish, and you deserve every bit of this warmth. Let’s make this happiness shine even brighter!",
                        "Your happiness about {reason} is absolutely contagious, and I’m so excited for you! It’s amazing to see you glowing, especially as a young adult navigating life’s ups and downs. Let’s keep this joy flowing!"
                    ],
                    "steps": [
                        "🌞 Savor the Moment: Take a deep breath and really feel this happiness—notice where it lives in your body, maybe your chest or a big smile. Pause for 30 seconds to soak it in. If you want, snap a quick photo or note to remember this feeling.",
                        "🧠 Reflect on the Joy: Ask yourself, 'What’s making me so happy right now?' Maybe it’s a specific moment or feeling. Say to yourself, 'I’m happy because of {reason}.' Naming it helps you hold onto it longer.",
                        "🎉 Share the Positivity: Spread your joy—text a friend something uplifting, share a funny meme, or tell someone you appreciate them. Happiness grows when you share it, even in small ways.",
                        "💡 Amplify the Good: Do something that keeps this vibe going. Listen to an upbeat song, dance for a minute, or plan a fun activity for today, like watching a comedy or going for a walk. Keep the energy high!",
                        "🫂 Practice Gratitude: Write down 3 things you’re grateful for today—it could be this moment, a person, or something small. Gratitude locks in happiness and makes it last.",
                        "🌟 Plan for More Joy: Think of one way to bring more happiness tomorrow—like scheduling a fun outing or setting a goal to do something you love. You’re building a life full of these moments."
                    ],
                    "relationship_steps": [
                        "🌞 Cherish the Connection: Take a moment to feel the warmth of this relationship joy—maybe it’s a smile or a flutter in your heart. Close your eyes for 20 seconds and let it sink in. If you’re with your partner, give them a quick hug or smile to share it.",
                        "🧠 Celebrate the Moment: Ask, 'What’s so special about this relationship right now?' Maybe it’s feeling understood or loved. Say, 'I’m happy because my relationship feels so good.' Naming it makes it even sweeter.",
                        "🎉 Share the Love: Do something small to spread this joy with your partner or a friend—send a sweet text, plan a fun date, or share a kind word. If you’re not with them, tell a friend about this happy moment to keep the vibe alive.",
                        "💡 Keep the Spark: Plan a little activity to keep this happiness flowing—like a movie night with your partner, a walk together, or even a silly dance session. If you’re flying solo, treat yourself to something fun, like a favorite treat.",
                        "🫂 Gratitude for Love: Write down 3 things you’re grateful for in this relationship (e.g., trust, laughter, support). Or just think of them and smile. Gratitude deepens your connection and joy.",
                        "🌟 Build More Happiness: Think of one way to nurture this relationship tomorrow—maybe a kind gesture, a heartfelt chat, or setting a boundary that keeps things healthy. You’re creating more moments like this."
                    ],
                    "follow_ups": [
                        "Want to plan a fun activity to keep this happiness going, like a special outing? Or share more about what’s making you smile? (Type 'plan', 'share', or your thoughts.)",
                        "How about deepening this joy with a gratitude practice or connecting with someone special? (Type 'gratitude', 'connect', or share more.)",
                        "Feeling inspired to set a goal for more joyful moments, like trying something new? Or we can explore another idea. (Type 'goal', 'explore', or your thoughts.)",
                        "Would you like to celebrate this moment with a creative idea, like journaling about your happiness? (Type 'creative', 'journal', or share more.)"
                    ]
                },
                "anxious": {
                    "intros": [
                        "I hear how anxious you’re feeling about {reason}. It’s completely understandable to feel overwhelmed, but remember that this feeling is temporary and you are safe right now.",
                        "Anxiety about {reason} can be so draining. I'm here with you. Take a moment to just be here, and we'll get through this together."
                    ],
                    "steps": [
                        "🌿 Breathe: Try box breathing. Inhale for 4 seconds, hold for 4, exhale for 4, and hold for 4. Repeat this 3 times.",
                        "🧠 Grounding: Look around and find 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                        "✍️ Release: Write down your biggest worry right now, and then write one reason why it might not happen or why you can handle it if it does.",
                        "💡 Focus: Do a simple, repetitive task like washing dishes, folding laundry, or doodling to give your mind a break.",
                        "🫂 Self-Compassion: Remind yourself out loud: 'I am doing the best I can, and that is enough.'",
                        "🌟 Plan: Break down what's making you anxious into small, manageable steps. Focus only on the very first step."
                    ],
                    "follow_ups": [
                        "Would you like to try another breathing exercise or talk more about what's making you anxious? (Type 'breathe', 'talk', or share more.)",
                        "Want to explore some CBT techniques to reframe these anxious thoughts? (Type 'cbt' or your thoughts.)",
                        "I’m here if you want to create a step-by-step plan to tackle this. (Type 'plan' or share more.)",
                        "Would you prefer to do a guided visualization to find a calm space? (Type 'creative', 'visualize' or your thoughts.)"
                    ]
                },
                "angry": {
                    "intros": [
                        "I can hear how angry you are about {reason}, and your anger is completely valid. It’s okay to feel this way when things are unfair or frustrating.",
                        "It sounds like you're furious about {reason}, and I want you to know I'm listening. Anger is a natural response, and you don't have to carry it alone."
                    ],
                    "steps": [
                        "🌿 Cool Down: Step away from the situation if you can. Splash cold water on your face or hold an ice cube—the temperature change can help reset your nervous system.",
                        "🧠 Name It: Say out loud, 'I am angry because...' Validating your own emotion takes away some of its explosive power.",
                        "✍️ Vent Safely: Write down everything you're feeling on a piece of paper, without filtering it. When you're done, rip it up and throw it away to symbolize letting go.",
                        "💡 Move Your Body: Do 10 jumping jacks, punch a pillow, or go for a brisk walk. Physical movement gives the adrenaline somewhere to go.",
                        "🫂 Empathy Pause: Once you're calmer, try to look at the situation from an outside perspective, not to excuse it, but to find clarity.",
                        "🌟 Constructive Action: Decide on one productive step you can take tomorrow to address the cause of your anger, like setting a boundary."
                    ],
                    "follow_ups": [
                        "Would you like to vent more about this, or should we try a physical exercise to release the tension? (Type 'talk', 'exercise', or share more.)",
                        "Want to explore a CBT trick for managing anger in the moment? (Type 'cbt' or your thoughts.)",
                        "I’m here if you want help drafting a constructive message or setting a boundary. (Type 'plan' or share more.)"
                    ]
                },
                "stressed": {
                    "intros": [
                        "You sound so stressed about {reason}. I know how heavy and suffocating that pressure can feel, but we can take this one step at a time.",
                        "I'm sorry the stress of {reason} is weighing on you so much. It's okay to feel overwhelmed, and you're doing your best."
                    ],
                    "steps": [
                        "🌿 Hit Pause: Close your eyes and take 3 deep belly breaths. Tell yourself, 'I am doing the best I can right now.'",
                        "🧠 Brain Dump: Write down every single thing you have to do or everything that's stressing you out. Getting it out of your head makes it less intimidating.",
                        "✍️ Prioritize: Pick just ONE thing from that list to focus on right now. Ignore the rest for the next 30 minutes.",
                        "💡 Set a Boundary: Say 'no' to one extra request today, or give yourself permission to lower your standards on a non-essential task.",
                        "🫂 Self-Care: Take a 10-minute break away from screens. Drink some water or make a cup of tea.",
                        "🌟 Big Picture: Remind yourself of a time you handled a stressful situation in the past. You survived that, and you will get through this too."
                    ],
                    "follow_ups": [
                        "Want to try a grounding exercise to bring your stress levels down? (Type 'exercise', 'mindfulness' or share more.)",
                        "Would you like to talk through your to-do list so we can organize it together? (Type 'plan', 'talk', or your thoughts.)",
                        "I'm here if you just want to take a moment to breathe. (Type 'breathe' or share more.)"
                    ]
                },
                "frustrated": {
                    "intros": [
                        "It makes complete sense that you're frustrated by {reason}. When things don't go as planned or feel stuck, it’s really hard.",
                        "I hear your frustration about {reason}. Feeling blocked or misunderstood is exhausting, and I'm here to support you."
                    ],
                    "steps": [
                        "🌿 Take a Step Back: Give yourself permission to walk away from the problem for 15 minutes. A fresh perspective needs a little distance.",
                        "🧠 Acknowledge the Block: Remind yourself that frustration often comes right before a breakthrough or a necessary change in direction.",
                        "✍️ Shift Focus: Do a tiny task you know you can finish easily—like organizing your desk or making the bed—to get a quick win.",
                        "💡 Reframe: Ask yourself, 'Is there another way to look at this problem, or a different approach I haven't tried?'",
                        "🫂 Vent it Out: Talk to someone who will just listen without trying to fix it immediately.",
                        "🌟 Acceptance: Accept that some things are out of your control right now, and focus on what you *can* control, even if it's small."
                    ],
                    "follow_ups": [
                        "Would you like to try a creative exercise to take your mind off things? (Type 'creative', 'explore' or share more.)",
                        "Want to talk more about what exactly is causing the frustration? (Type 'talk' or your thoughts.)",
                        "We can brainstorm some new approaches together if you're ready. (Type 'plan' or share more.)"
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise Exception(f"Failed to initialize SerenitySupport: {str(e)}")

    def analyze_mood_and_tone(self, user_input: str, user_id: str) -> Dict[str, any]:
        """Analyze user input for mood, tone, and context."""
        try:
            profile = self.user_profiles.get(user_id, {"last_input": "", "moods": [], "tone": "", "context": []})
            if profile["last_input"] == user_input and user_input:
                return {"moods": profile["moods"], "tone": profile["tone"], "context": profile["context"]}

            detected_moods = [mood for mood, keywords in self.mood_keywords.items() if any(keyword in user_input.lower() for keyword in keywords)]

            detected_tone = "neutral"
            for tone, keywords in self.tone_keywords.items():
                if any(keyword in user_input.lower() for keyword in keywords):
                    detected_tone = tone
                    break

            detected_context = [ctx for ctx, keywords in self.context_keywords.items() if any(keyword in user_input.lower() for keyword in keywords)]

            if not detected_moods or not detected_context:
                prompt = (
                    f"Analyze the input to identify primary and secondary emotions (e.g., happy, sad, anxious), "
                    f"emotional tone (e.g., positive, negative, neutral), and context (e.g., exam_stress, social). "
                    f"Return a JSON object with 'moods' as a list, 'tone' as a string, and 'context' as a list. "
                    f"Input: {user_input}"
                )
                try:
                    response = self.model.generate_content(prompt)
                    # Clean up response to ensure valid json
                    text = response.text.strip()
                    if text.startswith("```json"): text = text[7:]
                    if text.startswith("```"): text = text[3:]
                    if text.endswith("```"): text = text[:-3]
                    text = text.strip()
                    
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        result = eval(text)
                        
                    detected_moods = result.get("moods", detected_moods or ["uncertain"])
                    detected_tone = result.get("tone", detected_tone or "neutral")
                    detected_context = result.get("context", detected_context or [])
                except Exception as e:
                    logger.warning(f"Gemini analysis failed: {e}")

            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "moods": [], "last_input": "", "used_responses": [],
                    "context": [], "tone": "", "preferences": {}, "age": None
                }
            self.user_profiles[user_id].update({
                "moods": detected_moods,
                "last_input": user_input,
                "tone": detected_tone,
                "context": detected_context
            })

            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            self.conversation_history[user_id].append({
                "input": user_input,
                "moods": detected_moods,
                "tone": detected_tone,
                "context": detected_context,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "response": ""
            })

            return {"moods": detected_moods, "tone": detected_tone, "context": detected_context}
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {"moods": ["uncertain"], "tone": "neutral", "context": []}

    def generate_response(self, user_input: str, user_id: str, age: Optional[int] = None, reason: Optional[str] = None, conversation_id: Optional[str] = None, stop: bool = False) -> Dict[str, any]:
        """Generate a comprehensive, empathetic response with detailed mental health guidance."""
        try:
            if conversation_id and stop:
                if conversation_id in self.conversation_state:
                    del self.conversation_state[conversation_id]
                return {"response": "Thank you for talking with me. I'm here if you need anything else.", "conversation_id": None}

            # Check if user input matches a special follow-up action (fuzzy matching)
            matched_special = None
            if user_input:
                user_input_lower = user_input.lower()
                for key in self.special_inputs:
                    if key in user_input_lower:
                        matched_special = key
                        break

            if conversation_id and conversation_id in self.conversation_state:
                # Continue conversation
                state = self.conversation_state[conversation_id]
                mood = state["mood"]
                used_responses = state.get("used_responses", [])

                if matched_special:
                    response_parts = self.special_inputs[matched_special]["response"]
                    follow_up = self.special_inputs[matched_special]["follow_up"]
                    final_response = "\n".join(response_parts) + "\n" + follow_up
                    return {"response": final_response, "conversation_id": conversation_id}

                # If the user responds with something else, use Gemini to acknowledge it gracefully
                try:
                    ack_prompt = f"The user previously expressed feeling '{mood}'. They just said: '{user_input}'. " \
                                 f"Respond with a very brief, empathetic acknowledgment (1-2 sentences). " \
                                 f"Do not give advice or ask a follow up question, just acknowledge what they said."
                    gemini_ack = self.model.generate_content(ack_prompt)
                    ack_text = gemini_ack.text.strip()
                except Exception as e:
                    logger.warning(f"Failed to generate acknowledgement: {e}")
                    ack_text = "I hear you, and I appreciate you sharing that with me."

                # Generate a new follow-up question
                components = self.mood_library.get(mood, {})
                if not components:
                    if conversation_id in self.conversation_state:
                        del self.conversation_state[conversation_id]
                    return {"response": f"{ack_text} I'm here to listen if you want to talk about something else.", "conversation_id": None}

                available_follow_ups = [f for f in components.get("follow_ups", []) if f not in used_responses]
                if not available_follow_ups:
                    if conversation_id in self.conversation_state:
                        del self.conversation_state[conversation_id]
                    return {"response": f"{ack_text} We've covered a lot on this topic. Is there something else you'd like to explore?", "conversation_id": None}

                follow_up = random.choice(available_follow_ups)
                used_responses.append(follow_up)
                self.conversation_state[conversation_id]["used_responses"] = used_responses
                return {"response": f"{ack_text}\n\n{follow_up}", "conversation_id": conversation_id}
            else:
                # New conversation
                conversation_id = f"conv_{uuid4()}"
                analysis = self.analyze_mood_and_tone(user_input, user_id)
                moods, tone, context = analysis["moods"], analysis["tone"], analysis["context"]
                mood = moods[0] if moods else "uncertain"
                profile = self.user_profiles.get(user_id, {})
                used_responses = profile.get("used_responses", [])
                if age is not None:
                    profile["age"] = age

                response_parts = []
                follow_up = ""

                if matched_special:
                    response_parts = self.special_inputs[matched_special]["response"]
                    follow_up = self.special_inputs[matched_special]["follow_up"]
                else:
                    if mood in self.mood_library:
                        components = self.mood_library[mood]
                        intro = random.choice([i for i in components["intros"] if i not in used_responses] or components["intros"])
                        intro = intro.format(reason=reason or "this moment")
                        response_parts.append(intro)

                        if "social" in context and reason and "relationship" in reason.lower():
                            steps = components.get("relationship_steps", components["steps"])
                        else:
                            steps = components["steps"]

                        available_steps = [s for s in steps if s not in used_responses]
                        selected_steps = random.sample(available_steps, min(5, len(available_steps))) if available_steps else random.sample(steps, min(5, len(steps)))
                        response_parts.append("\nHere’s a guide to keep your heart glowing:\n" if mood == "happy" else "\nHere’s a gentle guide to help you navigate this moment:\n")
                        for i, step in enumerate(selected_steps, 1):
                            step = step.format(reason=reason or "this moment")
                            response_parts.append(f"Step {i}: {step}")
                            used_responses.append(step)

                        available_follow_ups = [f for f in components["follow_ups"] if f not in used_responses]
                        follow_up = random.choice(available_follow_ups or components["follow_ups"])
                        used_responses.append(follow_up)
                    else:
                        age_context = "a young adult" if age and 18 <= age <= 25 else "someone navigating their emotions"
                        tone_instruction = "uplifting and celebratory" if mood == "happy" else "compassionate and validating"
                        prompt = (
                            f"You are SerenityBot, responding in a {tone_instruction} tone. The user, {age_context}, is feeling '{mood}' because of '{reason or 'unknown'}'. "
                            f"Input: '{user_input}'. Context: {', '.join(context) or 'none'}. "
                            f"Generate a response starting with a heartfelt intro (e.g., 'I’m so thrilled...' for happy, 'I'm really sorry...' for others). "
                            f"Follow with 5 numbered steps (use emojis like 🌞, 🎉 for happy; 🌿, 🧠 for others) offering immediate and long-term mental health strategies (e.g., gratitude, sharing joy for happy; grounding, self-care for others) tailored to the mood and reason. "
                            f"End with 3 diverse follow-up options. Use an empowering tone."
                        )
                        try:
                            gemini_response = self.model.generate_content(prompt)
                            response_parts.append(gemini_response.text)
                            follow_up = f"Would you like to {'plan a joyful activity' if mood == 'happy' else 'try a mindfulness exercise'}, create a {'joy plan' if mood == 'happy' else 'self-care plan'}, or {'connect with others' if mood == 'happy' else 'find support resources'}? (Type {'plan' if mood == 'happy' else 'mindfulness'}, {'plan' if mood == 'happy' else 'plan'}, {'connect' if mood == 'happy' else 'resources'}, or share more.)"
                        except Exception as e:
                            logger.warning(f"Gemini response failed: {e}")
                            if mood == "happy":
                                response_parts.append(
                                    "I’m so thrilled you’re feeling happy—it’s such a beautiful moment!\n"
                                    "Here’s a guide to keep your heart glowing:\n"
                                    "Step 1: 🌞 Pause and feel this joy in your body.\n"
                                    "Step 2: 🧠 Name what’s making you happy.\n"
                                    "Step 3: 🎉 Share it with a friend or loved one.\n"
                                    "Step 4: 💡 Do something fun, like dancing.\n"
                                    "Step 5: 🫂 Write 3 things you’re grateful for."
                                )
                                follow_up = "Want to plan a fun activity or share more? (Type 'plan', 'share', or your thoughts.)"
                            else:
                                response_parts.append(
                                    "I'm really sorry you're feeling this way—it’s tough, and you’re not alone.\n"
                                    "Here’s a gentle guide to help you today:\n"
                                    "Step 1: 🌿 Take 5 deep breaths to calm your body.\n"
                                    "Step 2: 🧠 Name one feeling, like ‘I’m sad.’\n"
                                    "Step 3: ✍️ Write one thought to release it.\n"
                                    "Step 4: 💡 Ground yourself: Name 5 things you see.\n"
                                    "Step 5: 🫂 Do one kind thing, like resting."
                                )
                                follow_up = "Want to try a calming exercise or talk more? (Type 'exercise', 'talk', or your thoughts.)"

                final_response = "\n".join(response_parts) + "\n" + follow_up
                if profile:
                    profile["used_responses"] = used_responses[-50:]

                self.conversation_state[conversation_id] = {
                    "mood": mood,
                    "used_responses": used_responses,
                    "user_id": user_id,
                }

                self.conversation_history.setdefault(user_id, []).append({
                    "input": user_input,
                    "response": final_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                logger.info(f"User: {user_input}\nResponse: {final_response}")
                return {"response": final_response, "conversation_id": conversation_id}

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {"response": "I'm having trouble right now. Please try again later.", "conversation_id": None}

# Initialize SerenitySupport instance
serenity = SerenitySupport()

@app.post("/generate_response", summary="Generate an empathetic response based on user input")
async def generate_response(user_input: UserInput):
    """Generate a SerenityBot response based on user input."""
    try:
        # Validate input
        if not user_input.mood:
            raise HTTPException(status_code=400, detail="Mood is required.")

        # Generate or use provided user_id
        user_id = user_input.user_id or f"user_{uuid4()}"

        # Construct input text
        if user_input.input_text:
            input_text = user_input.input_text
        else:
            input_text = f"I’m feeling {user_input.mood} because {user_input.reason}" if user_input.reason else f"I’m feeling {user_input.mood}"

        # Generate response
        response_data = serenity.generate_response(
            user_input=input_text,
            user_id=user_id,
            age=user_input.age,
            reason=user_input.reason,
            conversation_id=user_input.conversation_id,
            stop=user_input.stop
        )

        # Return JSON response
        return {
            "user_id": user_id,
            "input": {
                "mood": user_input.mood,
                "age": user_input.age,
                "reason": user_input.reason,
                "input_text": input_text
            },
            "response": response_data["response"],
            "conversation_id": response_data["conversation_id"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/", summary="API root endpoint")
async def root():
    """Return a welcome message."""
    return {"message": "Welcome to SerenityBot API! Use POST /generate_response to interact."}