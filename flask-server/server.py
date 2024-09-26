from flask import Flask, request, jsonify
import json

#CODE BERT
import torch
from transformers import RobertaTokenizer, RobertaModel
# Load the CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

#For new best matching
import spacy
nlp = spacy.load("en_core_web_sm")


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

import os
import re



genai.configure(api_key=os.environ["GEMINI_API_KEY"])


model = genai.GenerativeModel('gemini-1.5-flash')

# response = model.generate_content("What is my name ? ")
# print(response.text)
# response = model.generate_content("What is my age ? ")
# print(response.text)
# response = model.generate_content("What did I study ? ")
# print(response.text)



app = Flask(__name__)

# Members API Route



analysisData = []


def read_folder_content(folder_path, output_file):
    print("LOG : Reading files")

    """Reads all files in a folder and its subfolders recursively, returning a single string without line breaks.

    Args:
        folder_path (str): The path to the folder to read.

    Returns:
        str: A string containing the concatenated content of all files.
    """

    content = ""
    currentAnalysis = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.startswith('.') and not file.endswith('.gz') and not file.endswith('.zip'):  # Ignore hidden files
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    if not file_path.startswith(os.path.join(folder_path, 'node_modules')):
                        try:
                            content = f.read().replace("\n", "")  # Remove line breaks
                            parent_folder = os.path.basename(os.path.dirname(f.name))
                            currentAnalysis += f"{parent_folder}/{file}: {content}"  # Append filename and content
                            #content += f.read().replace("\n", "")  # Remove line breaks

                        except UnicodeDecodeError as e:
                            print(f"Error decoding {file_path}: {e}")
                        # Store analysis after 5 files
            if len(files) > 0 and files.index(file) % 2 == 1:
                #response = model.generate_content("I want you to generate a json kind of format from a given codebase. it should have each and every details of the codebase in deep. later I will provide you this codebase and you should be able to write the code itself from it. but the reason to convert to json is to store it in some database. so try to make it short as much as possible without loosing any info from it.I asked you the same in previous conversation but the response is incomplete or stuck. i got a truncated response. following is the code base : " + currentAnalysis)
                analysisData.append(currentAnalysis)
                currentAnalysis = ""  # Reset for next analysis set
            else:
                # Handle hidden files (optional)
                # You can add logic here to handle hidden files,
                # but it's skipped by default.
                pass
            
        # Handle remaining files if less than 5
        if currentAnalysis:
            #response = model.generate_content("I want you to generate a json kind of format from a given codebase. it should have each and every details of the codebase in deep. later I will provide you this codebase and you should be able to write the code itself from it. but the reason to convert to json is to store it in some database. so try to make it short as much as possible without loosing any info from it.I asked you the same in previous conversation but the response is incomplete or stuck. i got a truncated response. following is the code base : " + currentAnalysis)
            analysisData.append(currentAnalysis)

    # Handle remaining files if less than 5
    if currentAnalysis:
        response = model.generate_content("I want you to generate a json kind of format from a given codebase. it should have each and every details of the codebase in deep. later I will provide you this codebase and you should be able to write the code itself from it. but the reason to convert to json is to store it in some database. so try to make it short as much as possible without loosing any info from it.I asked you the same in previous conversation but the response is incomplete or stuck. i got a truncated response. following is the code base : " + currentAnalysis)
        analysisData.append(currentAnalysis)

    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write(content)

    return content

# Replace "e-commerce-react" with the actual path to your folder
folder_path = "/Users/mohammedanasm/CodeRepo/eCommerceApp/ecommerce-react/src"
output_file = "combined_code.txt"
combined_content = read_folder_content(folder_path, output_file)
#print(analysisData)

# def find_best_analysis(user_query):
#     # Preprocess user query and analyzed data
#     stop_words = set(stopwords.words('english'))
#     def preprocess(text):
#         words = nltk.word_tokenize(text.lower())
#         words = [word for word in words if word not in stop_words]
#         return " ".join(words)

#     analysisDataJson = (analysisData)

#     user_query_processed = preprocess(user_query)
#     analyzed_data_processed = [preprocess(analysis) for analysis in analysisDataJson]

#     # Vectorize data
#     vectorizer = TfidfVectorizer()
#     query_vector = vectorizer.fit_transform([user_query_processed])
#     analysis_vectors = vectorizer.transform(analyzed_data_processed)

#     # Calculate similarity
#     similarity_scores = cosine_similarity(query_vector, analysis_vectors)[0]
    
#     import heapq
#     top_3_indices = heapq.nlargest(3, range(len(similarity_scores)), key=similarity_scores.__getitem__)

#     # Return the corresponding analyses (modify to fit your needs)
#     top_3_analyses = [analysisDataJson[index] for index in top_3_indices]
#     return top_3_analyses

#     # Find best match
#     # best_match_index = similarity_scores.argmax()
#     # best_analysis = analysisData[best_match_index]

#     # return best_analysis

def find_best_analysis(user_query):
    print("LOG : Analysis started")
    # Load NLP model
    nlp = spacy.load("en_core_web_sm")

    # Preprocess user query and analyzed data
    stop_words = set(stopwords.words('english'))
    def preprocess(text):
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    analysisDataJson = (analysisData)

    user_query_processed = preprocess(user_query)
    analyzed_data_processed = [preprocess(analysis) for analysis in analysisDataJson]

    # Tokenize and embed
    user_query_doc = nlp(user_query_processed)
    analyzed_data_docs = [nlp(analysis) for analysis in analyzed_data_processed]

    # Calculate similarity
    similarity_scores = [user_query_doc.similarity(doc) for doc in analyzed_data_docs]

    # Return the top 3 analyses
    top_10_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:50]
    top_10_analyses = [analysisDataJson[index] for index in top_10_indices]
    return top_10_analyses
#              {product.price ? displayMoney(product.price) : <Skeleton width={40} />}

#prompt = "Where is the component for custom input for mobile in milk related section ?"

@app.route("/members", methods=["POST"])
def members():
    print("LOG : API HITTED")
    prompt = request.get_json()
    print("Prompt " , prompt)

    if prompt is None:
        return jsonify({"error": "Missing request body"}), 400  # Bad request
    #response = model.generate_content("If i want to provide you an entire code base, how do i achieve it ? This is a free git open repo, try helping me on the query what is there in 'Contact Manager app' in following repo : https://github.com/smthari/Frontend-Projects.git")
    #response = model.generate_content(prompt)
    #print(response.text)
    best_analysis = find_best_analysis(prompt)
    print("BEST ANALYSIS****************")
    print(best_analysis[0])
    responses = ''
    for analysis in best_analysis:
        #responses += model.generate_content(prompt + ". Below is the code base  " + analysis).text
        responses += "Snippet : " + analysis + "\n"
        
    #response = model.generate_content("Analyze your response and combine them properly with the information it has. they are three different responses from you. So combine them with proper information and removing duplicates. : " + responses)
    print("Responses given to model " + responses)
    response = model.generate_content("Analyze the below snippets. Also use the filenames and folder names to understand the snippets only for you. No need to provide file names and folder names in response. Utilize the filenames and folder names to answer the following question. Then answer the following question in the context of below code snippets : " + prompt + ". \n Following is the code snippets for you to analyze : " + responses + ". \n")
    #response = model.generate_content(prompt + ". Use this codebase for finding this answer. " + best_analysis[0])
    #print(response.text)

    return {"members":response.text}

if __name__ == "__main__":
    app.run(debug=True)