import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openrouter/free"

def spell_correction(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: '{query}'""",
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    corrected_query = content.strip().strip('"') if content else query
    return corrected_query

def rewrite_query(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """,
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    rewritten_query = content.strip().strip('"') if content else query
    return rewritten_query

def expand_query(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Expand the user-provided movie search query below with related terms.
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """,
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    expanded_query = content.strip().strip('"') if content else query
    return expanded_query

def rerank_individual(query: str, doc: dict) -> float:
    messages = [
        {
            "role": "user",
            "content": f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:""",
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    score = float(response.choices[0].message.content)
    return score

def rerank_batch(query: str, docs: list[dict]) -> list:

    doc_list_str = "\n".join(
        [f"{doc.get('id', '')}: {doc.get('title', '')} - {doc.get('description', '')}" for doc in docs]
    )   

    messages = [
        {
            "role": "user",
            "content": f"""Rank the movies listed below by relevance to the following search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return the movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:"""
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    ranked_ids = json.loads(content.strip()) if content else []
    new_dict = {doc['id']: doc for doc in docs}
    ranked_ids = [doc_id for doc_id in ranked_ids if doc_id in new_dict]
    final_ranking = []
    for index, doc_id in enumerate(ranked_ids):
        new_dict[doc_id]['rerank_rank'] = index + 1
        final_ranking.append(new_dict[doc_id])

    return final_ranking

def rerank_cross_encoder(query: str, docs: list[dict]) -> list:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    pairs = []
    for doc in docs:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    scores = cross_encoder.predict(pairs)
    for i in range(len(scores)):
        docs[i]["cross_encoder_score"] = scores[i]

   
    return sorted(docs, key=lambda x: x['cross_encoder_score'], reverse=True)