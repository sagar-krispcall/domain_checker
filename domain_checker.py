import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ----------------------------
# Load API key from .env file
# ----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# Configure Gemini API
genai.configure(api_key=api_key)

# ----------------------------
# Load your dataset
# ----------------------------
converted = pd.read_csv('C:/SAG/Air-Channel/files(Oct21-21)/converted_users.csv')

# ----------------------------
# Function to call Gemini API
# ----------------------------
def call_gemini(prompt):
    """
    Calls Gemini with the given prompt and returns the response text.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ----------------------------
# Enrich dataset automatically
# ----------------------------
for index, row in converted.iterrows():
    domain = row['domain']

    prompt = f"""
    Domain: {domain}
    Please identify:
    1. Company business category (e.g., Software, Retail, Marketing)
    2. Estimated employee size (e.g., 1-10, 11-50, 51-200, 201-500, 500+)
    Provide output strictly as JSON:
    {{
      "domain": "{domain}",
      "category": "...",
      "company_size": "..."
    }}
    """

    try:
        response_text = call_gemini(prompt)
        data = json.loads(response_text)  # Parse JSON from model
        converted.at[index, 'category'] = data.get('category', 'Unknown')
        converted.at[index, 'company_size'] = data.get('company_size', 'Unknown')
    except json.JSONDecodeError:
        print(f"⚠️ JSON parse error for {domain}: {response_text}")
        converted.at[index, 'category'] = 'Unknown'
        converted.at[index, 'company_size'] = 'Unknown'
    except Exception as e:
        print(f"⚠️ Error processing {domain}: {e}")
        converted.at[index, 'category'] = 'Unknown'
        converted.at[index, 'company_size'] = 'Unknown'

    # Sleep between requests to respect rate limits
    time.sleep(10)

# ----------------------------
# Save enriched dataset
# ----------------------------
converted.to_csv('C:/SAG/Air-Channel/files(Oct21-21)/converted_users_enriched.csv', index=False)

print("✅ Enrichment complete! Saved to converted_users_enriched.csv")
