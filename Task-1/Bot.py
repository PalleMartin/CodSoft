responses = {
    "hello": "Hello! How can I help you?",
    "hi": "Hi there!",
    "how are you": "I'm just a bot, but I'm doing great!",
    "name": "I am a simple chatbot.",
    "what can you do": "I can answer basic questions and have simple conversations.",
    "help": "You can ask me basic questions.",
    "find": "Sorry, I didn't catch that.",
    "bye": "Goodbye!"
    
}

print("Chatbot started (type 'bye' to exit)")

while True:

    user_input = input("You: ").lower()

    found = False

    for key in responses:
        if key in user_input:
            print("Bot:", responses[key])
            found = True
            break

    if not found:
        print("Bot: Sorry I don't understand.")

    if "bye" in user_input:
        break