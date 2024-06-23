import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import OpenAI as LangChainOpenAI  # この行を変更
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import logging

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
        
        self.chunks = [item['text'] for item in self.data]
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

    def vector_search(self, query, k=3):
        query_embedding = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        results = [self.chunks[i] for i in top_indices]
        return "\n\n".join(results)

    def keyword_search(self, query, k=3):
        # キーワードの抽出
        keywords = self.extract_keywords(query)
        
        # キーワードの埋め込みを生成
        keyword_embeddings = client.embeddings.create(
            input=keywords,
            model="text-embedding-3-small"
        ).data

        # 各キーワードの埋め込みと文書チャンクの埋め込みの類似度を計算
        similarities = np.mean([cosine_similarity([ke.embedding], self.embeddings)[0] for ke in keyword_embeddings], axis=0)
        
        top_indices = similarities.argsort()[-k:][::-1]
        results = [self.chunks[i] for i in top_indices]
        return "\n\n".join(results)

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

    def generate_model_response(self, model_name, query, search_results, discussion_topic):
        if model_name == "Model 1 (Vector Search)":
            system_prompt = f"""You are an AI assistant that uses vector search to retrieve information from academic papers. 
            Your task is to discuss the following topic: "{discussion_topic}"
            Use the provided search results to inform your discussion. Focus on how vector search has helped you find relevant information."""
        else:
            system_prompt = f"""You are an AI assistant that uses keyword search to retrieve information from academic papers. 
            Your task is to discuss the following topic: "{discussion_topic}"
            Use the provided search results to inform your discussion. Focus on how keyword search has helped you find relevant information."""

        prompt = f"""Based on the following search results from research papers, provide your perspective on the topic: '{discussion_topic}'.

        Search Query: {query}

        Search Results:
        {search_results}

        {model_name}'s Response:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
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

        for round_num in range(rounds):
            logging.info(f"Starting round {round_num + 1} of {rounds}")
            vector_query = self.generate_search_query(discussion_topic)
            keyword_query = self.generate_search_query(discussion_topic)
            
            vector_results = self.vector_search(vector_query)
            keyword_results = self.keyword_search(keyword_query)
            
            model1_data = self.generate_model_response("Model 1 (Vector Search)", vector_query, vector_results, discussion_topic)
            model2_data = self.generate_model_response("Model 2 (Keyword Search)", keyword_query, keyword_results, discussion_topic)
            
            round_data = {
                "Model 1 (Vector Search)": {
                    "query": vector_query,
                    "data": model1_data
                },
                "Model 2 (Keyword Search)": {
                    "query": keyword_query,
                    "data": model2_data
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