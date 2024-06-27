import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
import random

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI クライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# キーワード抽出のための Pydantic モデル
class Keywords(BaseModel):
    keywords: List[str] = Field(description="List of extracted keywords")

class DiscussionGenerator:
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.chunks = [{"id": item['chunk_id'], "text": item['text']} for item in self.data]
        self.embeddings = [item['embedding'] for item in self.data]
        self.output_folder = "discussion_results"
        
        # LangChain用のLLMの初期化
        self.llm = LangChainOpenAI(temperature=0.1)

        # キーワード抽出用のパーサーの設定
        self.keyword_parser = PydanticOutputParser(pydantic_object=Keywords)

    def generate_search_query(self, topic):
        prompt = PromptTemplate(
            template="Given the topic: '{topic}', generate a specific search query that would be useful for finding relevant information in academic papers. The query should be focused and use technical terms related to the topic.\n\nSearch Query:",
            input_variables=["topic"]
        )

        _input = prompt.format_prompt(topic=topic)
        query = self.llm(_input.to_string())
        return query.strip()

    def vector_search(self, query, k=3) -> List[Dict[str, str]]:
        query_embedding = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        results = [{"id": self.chunks[i]["id"], "text": self.chunks[i]["text"]} for i in top_indices]
        return results

    def keyword_search(self, query, k=3) -> List[Dict[str, str]]:
        keywords = self.extract_keywords(query)
        
        matching_chunks = []
        for chunk in self.chunks:
            if any(keyword.lower() in chunk["text"].lower() for keyword in keywords):
                matching_chunks.append(chunk)
        
        if len(matching_chunks) < k:
            for chunk in self.chunks:
                if any(any(kw.lower() in word.lower() for word in chunk["text"].split()) for kw in keywords):
                    if chunk not in matching_chunks:
                        matching_chunks.append(chunk)
        
        selected_chunks = random.sample(matching_chunks, min(k, len(matching_chunks)))
        return [{"id": chunk["id"], "text": chunk["text"]} for chunk in selected_chunks]

    def extract_keywords(self, text):
        prompt = PromptTemplate(
            template="Extract the key technical terms, concepts, or proper nouns from the following text:\n\n{query}\n\n{format_instructions}",
            input_variables=["query"],
            partial_variables={"format_instructions": self.keyword_parser.get_format_instructions()}
        )

        _input = prompt.format_prompt(query=text)
        output = self.llm(_input.to_string())
        keywords = self.keyword_parser.parse(output).keywords
        return keywords

    def generate_model_response(self, model_name, query, search_results, discussion_topic, conversation_history, is_first_round):
        search_results_text = "\n\n".join([f"Chunk {result['id']}:\n{result['text']}" for result in search_results])
        
        if model_name == "Model 1 (Vector Search)":
            system_prompt = f"""You are an AI assistant that uses vector search to retrieve information from academic papers. 
            Your task is to discuss the following topic: "{discussion_topic}"
            Use the following search results to inform your discussion. Focus on how vector search has helped you find relevant information.
            
            Search Query: {query}

            Search Results:
            {search_results_text}

            Engage in a discussion with the other model, addressing its previous points if any."""
        else:
            system_prompt = f"""You are an AI assistant that uses keyword search to retrieve information from academic papers. 
            Your task is to discuss the following topic: "{discussion_topic}"
            Use the following search results to inform your discussion. Focus on how keyword search has helped you find relevant information.
            
            Search Query: {query}

            Search Results:
            {search_results_text}

            Engage in a discussion with the other model, addressing its previous points if any."""

        if is_first_round:
            user_prompt = f"Let's start our discussion on the topic: '{discussion_topic}'. Please provide your initial thoughts based on your search results."
        else:
            user_prompt = conversation_history[-1]  # 相手モデルの最後の返答

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response.choices[0].message.content
        }

    def generate_discussion(self, discussion_topic, rounds=1):
        logging.info(f"Generating discussion for topic: {discussion_topic}")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        discussion_data = {
            "topic": discussion_topic,
            "rounds": []
        }

        conversation_history = []

        for round_num in range(rounds):
            logging.info(f"Starting round {round_num + 1} of {rounds}")
            is_first_round = (round_num == 0)
            
            vector_query = self.generate_search_query(discussion_topic)
            keyword_query = self.generate_search_query(discussion_topic)
            
            vector_results = self.vector_search(vector_query)
            keyword_results = self.keyword_search(keyword_query)
            
            model1_data = self.generate_model_response("Model 1 (Vector Search)", vector_query, vector_results, discussion_topic, conversation_history, is_first_round)
            conversation_history.append(f"Model 1: {model1_data['response']}")
            
            model2_data = self.generate_model_response("Model 2 (Keyword Search)", keyword_query, keyword_results, discussion_topic, conversation_history, is_first_round)
            conversation_history.append(f"Model 2: {model2_data['response']}")
            
            round_data = {
                "Model 1 (Vector Search)": {
                    "query": vector_query,
                    "data": {
                        "system_prompt": model1_data["system_prompt"],
                        "user_prompt": model1_data["user_prompt"],
                        "response": model1_data["response"],
                        "used_chunk_ids": [result["id"] for result in vector_results]
                    }
                },
                "Model 2 (Keyword Search)": {
                    "query": keyword_query,
                    "data": {
                        "system_prompt": model2_data["system_prompt"],
                        "user_prompt": model2_data["user_prompt"],
                        "response": model2_data["response"],
                        "used_chunk_ids": [result["id"] for result in keyword_results]
                    }
                }
            }
            discussion_data["rounds"].append(round_data)

        # Save discussion results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"discussion_{timestamp}.json"
        filepath = os.path.join(self.output_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(discussion_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Discussion generated and saved to {filepath}")
        return discussion_data, filepath

# 使用例
if __name__ == "__main__":
    generator = DiscussionGenerator("example_processed.json")
    discussion, filepath = generator.generate_discussion("What are the latest advancements in RAG models?", rounds=2)
    print(f"Discussion saved to: {filepath}")
    print(json.dumps(discussion, indent=2))