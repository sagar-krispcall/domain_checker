import pandas as pd
import json
import itertools
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from datetime import datetime

# ------------ Load API Keys ------------
load_dotenv()
api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
]

api_keys = [k for k in api_keys if k]
if not api_keys:
    raise ValueError("❌ No valid Gemini API keys found in .env")

key_cycle = itertools.cycle(api_keys)

def configure_genai(key):
    genai.configure(api_key=key)

current_key = next(key_cycle)
configure_genai(current_key)
print(f"🔑 Using Gemini key: {current_key[:6]}...")

# ------------ Load CSV Data (as string-safe) ------------
file_path = r'C:/SAG/Air-Channel/files(Oct21-21)/converted_users.csv'
save_path = r'C:/SAG/Air-Channel/files(Oct21-21)/converted_users_enriched.csv'

df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

# Ensure required columns exist
for col in [
    'category',
    'company_size',
    'email_provider',
    'confidence',
    'other_industry_category',
    'website'
]:
    if col not in df.columns:
        df[col] = ""

# ------------ Gemini API Call Function ------------
def call_gemini(prompt, retries=3):
    global current_key
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            text = (response.text or "").strip()

            if not text:
                raise ValueError("Empty response from Gemini.")

            # Clean possible Markdown wrapping
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                else:
                    text = text.replace("```json", "").replace("```", "").strip()

            # Extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return match.group()
            else:
                raise ValueError(f"Could not extract JSON from response: {text}")

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                print(f"🚦 Quota limit hit for key {current_key[:6]}... Rotating key.")
                current_key = next(key_cycle)
                configure_genai(current_key)
                print(f"🔑 Switched to Gemini key: {current_key[:6]}...")
                time.sleep(60)
                continue

            elif "403" in err or "SERVICE_DISABLED" in err:
                print(f"🚫 API disabled for key {current_key[:6]}... Skipping key.")
                current_key = next(key_cycle)
                configure_genai(current_key)
                print(f"🔑 Switched to Gemini key: {current_key[:6]}...")
                continue

            elif "Empty response" in err:
                print("⚠ Empty response — waiting 20s then retrying...")
                time.sleep(20)
                continue

            else:
                raise RuntimeError(f"Gemini API error: {err}")

    raise RuntimeError("❌ All retries failed for Gemini API call.")

# ------------ Main Processing Loop ------------
start_time = datetime.now()
print(f"\n⏱ Start time: {start_time}\n")

processed = 0

try:
    for idx, row in df.iterrows():
        if row['category'] and row['category'] != "Unknown":
            continue

        domain = row.get('domain')
        if not domain:
            continue

        print(f"\n🌐 Processing: {domain}")

        # --- Updated Prompt ---
        prompt = f"""
        You are an enrichment API that classifies company domains.

        Given the domain below, return a single JSON object with the following fields:

        1. "domain": The same domain given as input.

        2. "category": One of these EXACT options:
            - Technology and software development
            - Marketing and advertising
            - E-commerce and retail
            - Financial Service
            - Education and e-learning
            - Real estate and property management
            - Healthcare
            - Logistics and transportation
            - Manufacturing and industrial
            - Other

        3. "other_industry_category":
            - Include this ONLY if "category" = "Other".
            - It should describe the most relevant industry type (for example: "Legal Services", "Travel Agency", "Non-profit", "Entertainment", etc.).
            - If "category" is not "Other", set this field to "N/A".

        4. "company_size": One of these EXACT options:
            - solo
            - 1-5 employees
            - 5-20
            - 20-50
            - 50-100
            - 100-200
            - 200-500
            - 500+

        5. "email_provider":
            Identify whether the domain is an email service provider rather than an actual business domain.
            Output:
            "Yes" → if the domain is used primarily for email services (free, business, or disposable).
            "No" → if the domain is a regular business or organization.
            Clarification:
            This variable exists because our system sometimes misclassifies email provider domains as business websites.

        6. "website":
            Determine how much usable business information is available on the domain's website.
            Output must be ONE of the following:
                - unreachable → The website doesn't load or shows errors, SSL issues, or blocks.
                - no_info → The site loads but has no real business info (e.g., placeholders or “coming soon”).
                - little_info → Some info exists but not enough to understand the business fully.
                - much_info → Clear and detailed business info available (services, products, About page, etc.).

        7. "confidence": A number between 0 and 100 representing confidence in the classification.

        Confidence Guidelines:
            - 90-100 → Strongly supported by verified or well-known online sources (e.g., official websites, LinkedIn, Crunchbase, business directories).
            - 70-89 → Partial or indirect evidence supporting classification (small websites, limited listings, etc.).
            - 40-69 → Weak or ambiguous information (minimal online presence or unclear company identity).
            - 0-39 → Very little or no reliable information found; classification is mostly a guess.

        Rules:
            - Respond ONLY with valid JSON (no markdown, no explanations).
            - Always include all keys, even if unsure (use "Unknown" or "N/A" if needed).
            - Output must start with '{{' and end with '}}'.

        Example output:
        {{
            "domain": "{domain}",
            "category": "Technology and software development",
            "other_industry_category": "N/A",
            "company_size": "20-50",
            "email_provider": "No",
            "website": "much_info",
            "confidence": 80
        }}

        Domain to analyze: {domain}
        """

        try:
            req_start = time.time()

            res = call_gemini(prompt)
            js = json.loads(res)

            req_time = round(time.time() - req_start, 2)

            print(f"📝 Gemini output for {domain}:")
            print(json.dumps(js, indent=2))
            print(f"⏱ Response time: {req_time} seconds")

            # --- Assign outputs ---
            df.at[idx, 'category'] = js.get('category', 'Unknown')
            df.at[idx, 'company_size'] = js.get('company_size', 'Unknown')
            df.at[idx, 'email_provider'] = js.get('email_provider', 'Unknown')
            df.at[idx, 'confidence'] = str(js.get('confidence', '0'))
            df.at[idx, 'other_industry_category'] = js.get('other_industry_category', 'N/A')
            df.at[idx, 'website'] = js.get('website', 'Unknown')

        except Exception as e:
            print(f"⚠ Error at {domain}: {e}")
            df.at[idx, 'category'] = "Unknown"
            df.at[idx, 'company_size'] = "Unknown"
            df.at[idx, 'email_provider'] = "Unknown"
            df.at[idx, 'confidence'] = "0"
            df.at[idx, 'other_industry_category'] = "N/A"
            df.at[idx, 'website'] = "Unknown"

        # --- Save progress safely ---
        df = df.astype(str)
        df.to_csv(save_path, index=False, quoting=1, encoding='utf-8', lineterminator='\n')
        processed += 1
        print(f"💾 Saved progress. Rows completed: {processed}")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving progress...")
    df = df.astype(str)
    df.to_csv(save_path, index=False, quoting=1, encoding='utf-8', lineterminator='\n')

# ------------ Completion Summary ------------
end_time = datetime.now()
elapsed = end_time - start_time

df = df.astype(str)
df.to_csv(save_path, index=False, quoting=1, encoding='utf-8', lineterminator='\n')

print("\n✅ Enrichment complete!")
print(f"⏱ Start time: {start_time}")
print(f"⏱ End time: {end_time}")
print(f"⏱ Total time taken: {elapsed}")
print(f"📁 Saved to: {save_path}")
