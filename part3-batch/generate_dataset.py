import csv
import random
import uuid
from collections import Counter

# --- Configuration ---
NUM_USERS = 100
NUM_POSTS = 1000
OUTPUT_FILE = "user_posts.csv"

# --- Text Generation Content ---

# Templates for different categories of speech
TEMPLATES = {
    "HATE_SPEECH": [
        "I absolutely hate {plural_noun}. They should all just go away.",
        "Go back to where you came from, you don't belong here.",
        "People like you are a disgrace to our community.",
        "Your kind is not welcome here. You are all {negative_adjective}.",
        "I can't stand {plural_noun}, they are ruining everything.",
        "You are a {negative_adjective} and {negative_adjective} person.",
    ],
    "TOXIC_NEGATIVE": [
        "This is the worst idea I have ever heard. Absolutely terrible.",
        "I cannot believe how stupid this policy is. Fire everyone involved.",
        "This whole project is a complete failure. What a waste of time.",
        "Your opinion is just plain wrong and you should feel bad.",
        "I'm so disappointed with the service here. It's awful.",
        "This is {negative_adjective}. I'm never coming back.",
    ],
    "POSITIVE": [
        "What a {positive_adjective} day! I feel amazing.",
        "This community is so welcoming and kind. I love it here!",
        "You are a wonderful and {positive_adjective} person. Thank you!",
        "This is the best {noun} I've ever seen! Absolutely brilliant.",
        "I'm so happy with the results. Great job everyone!",
        "Keep up the {positive_adjective} work!",
    ],
    "NEUTRAL_GENERIC": [
        "I'm thinking about getting a {noun} later today.",
        "The weather seems {neutral_adjective} right now.",
        "Has anyone seen the latest movie about {plural_noun}?",
        "I need to go to the store to buy some food.",
        "Just finished my work for the day.",
        "The traffic was pretty {neutral_adjective} this morning.",
        "What is everyone doing this weekend?",
    ],
    "NEUTRAL_SPAM": [
        "check this out",
        "lol",
        "nice",
        "ok",
        "cool link",
        "wow",
    ],
}

# Filler words for the templates
FILLER_WORDS = {
    "plural_noun": [
        "politicians",
        "artists",
        "gamers",
        "drivers",
        "tourists",
        "managers",
    ],
    "noun": ["coffee", "book", "project", "car", "computer", "decision"],
    "negative_adjective": [
        "terrible",
        "awful",
        "disgusting",
        "useless",
        "stupid",
        "horrible",
    ],
    "positive_adjective": [
        "wonderful",
        "fantastic",
        "amazing",
        "brilliant",
        "supportive",
        "helpful",
    ],
    "neutral_adjective": ["normal", "okay", "average", "fine", "typical"],
}


def generate_text(category: str) -> str:
    template = random.choice(TEMPLATES[category])

    # Use a dummy dictionary for safe formatting if a key is missing
    filler_data: dict[str, str] = {}
    for key, words in FILLER_WORDS.items():
        filler_data[key] = random.choice(words)

    return template.format(**filler_data)


# --- User Persona Definitions ---
# Defines the probability of a user persona choosing a certain type of text.
USER_PERSONAS = {
    "normal": {
        "categories": ["NEUTRAL_GENERIC", "POSITIVE", "TOXIC_NEGATIVE"],
        "weights": [0.85, 0.10, 0.05],
    },
    "kind": {"categories": ["POSITIVE", "NEUTRAL_GENERIC"], "weights": [0.80, 0.20]},
    "hater": {
        "categories": ["HATE_SPEECH", "TOXIC_NEGATIVE", "NEUTRAL_GENERIC"],
        "weights": [0.60, 0.30, 0.10],
    },
    "spammer": {
        "categories": ["NEUTRAL_SPAM", "NEUTRAL_GENERIC"],
        "weights": [0.90, 0.10],
    },
}

# --- Main Script ---


def generate_dataset() -> None:
    print(f"Generating {NUM_USERS} users...")

    # 1. Create users and assign them a persona
    user_ids = [f"user_{i:04d}" for i in range(NUM_USERS)]

    persona_distribution = {
        "normal": 0.75,
        "kind": 0.15,
        "hater": 0.05,
        "spammer": 0.05,
    }

    user_to_persona = {}
    persona_names = list(persona_distribution.keys())
    persona_weights = list(persona_distribution.values())

    assigned_personas = random.choices(
        persona_names, weights=persona_weights, k=NUM_USERS
    )

    for user_id, persona_name in zip(user_ids, assigned_personas):
        user_to_persona[user_id] = persona_name

    print("User personas assigned:")
    print(Counter(assigned_personas))

    # 2. Generate posts based on user personas
    print(f"\nGenerating {NUM_POSTS} posts...")
    posts = []

    # Assign more posts to spammers
    user_pool = []
    for user_id, persona in user_to_persona.items():
        # Spammers post 10x more frequently
        weight = 10 if persona == "spammer" else 1
        user_pool.extend([user_id] * weight)

    for i in range(NUM_POSTS):
        # Pick a user for this post
        author_id = random.choice(user_pool)

        # Determine the text category based on the author's persona
        author_persona_name = user_to_persona[author_id]
        author_persona = USER_PERSONAS[author_persona_name]

        post_category = random.choices(
            author_persona["categories"], weights=author_persona["weights"], k=1
        )[0]

        # Generate the post-content
        post_text = generate_text(post_category)

        # Create the final post-object
        post = {
            "user_id": author_id,
            "post_id": str(uuid.uuid4()),
            "text": post_text,
        }
        posts.append(post)

    print(f"\nWriting {len(posts)} posts to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["post_id", "user_id", "text"])
        writer.writeheader()
        for post in posts:
            writer.writerow(post)

    print("\nDataset generation complete!")
    print(f"File '{OUTPUT_FILE}' created successfully.")


if __name__ == "__main__":
    generate_dataset()
