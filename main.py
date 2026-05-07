import os

from dotenv import load_dotenv

load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_openrouter import ChatOpenRouter

if not os.getenv("OPENROUTER_API_KEY"):
    print("OPENROUTER_API_KEY is not set")
    exit(1)

model = ChatOpenRouter(
    model="anthropic/claude-sonnet-4.5",
    temperature=0,
    max_tokens=1024,
    max_retries=2,
    # other params...
)


def main():
    print("Hello from basics!")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)

print(ai_msg.content)
if __name__ == "__main__":
    main()
