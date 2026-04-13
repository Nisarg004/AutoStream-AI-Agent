import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_intent(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user message into exactly one of: "
                        "greeting, product_query, high_intent.\n"
                        "- greeting: hellos, hi, hey, thanks, casual conversation\n"
                        "- product_query: questions about pricing, features, plans, policies\n"
                        "- high_intent: user wants to sign up, buy, try, start, or subscribe\n"
                        "Return only the label, nothing else. No explanation."
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_tokens=10,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return _keyword_fallback(message)


def _keyword_fallback(message: str) -> str:
    message = message.lower()
    high_intent = [
        "sign up", "buy", "try", "subscribe", "pro plan",
        "basic plan", "want to", "start", "get started", "purchase",
    ]
    product_query = [
        "price", "plan", "feature", "cost", "support",
        "refund", "resolution", "caption", "video", "how much", "what is",
    ]

    if any(k in message for k in high_intent):
        return "high_intent"
    if any(k in message for k in product_query):
        return "product_query"
    return "greeting"
