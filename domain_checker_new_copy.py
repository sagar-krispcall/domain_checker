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

# ------------ Load CSV Data ------------
file_path = r'C:/SAG/Air-Channel/files(Oct21-21)/not_converted_users.csv'
save_path = r'C:/SAG/Air-Channel/files(Oct21-21)/not_converted_users_enriched.csv'

df = pd.read_csv(file_path)

# Ensure required columns exist
for col in ['category','company_size','email_provider','confidence','other_industry_category','website']:
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
        You are an enrichment API that classifies company domains. Your goal is to return a single JSON object with accurate classification and information. 

        Guidelines:

        1. "domain": Return the exact domain given as input.

        2. "category": Choose ONE of the following EXACT options:
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
            - Describe the most relevant industry type (e.g., "Legal Services", "Travel Agency", "Non-profit", "Entertainment").
            - If "category" is not "Other", set this to "N/A".

        4. "company_size": Choose ONE of these EXACT options:
            - solo
            - 1 to 5 employees
            - 5 to 20
            - 20 to 50
            - 50 to 100
            - 100 to 200
            - 200 to 500
            - 500+
            Guidelines:
                - Use LinkedIn, Crunchbase, official directories, or company reports to estimate size.
                - If exact number is unknown, provide the closest estimation.

        5. "email_provider": Identify if the domain is primarily used as an email service (free, business, disposable).
            - Output "Yes" if it is an email provider.
            - Output "No" if it is a regular business or organization.
            - Examples:
                - gmail.com → Yes
                - yahoo.com → Yes
                - company.com → No
            - If unsure, make your best guess based on domain reputation and online sources.

        6. "website": Evaluate the domain's website for usable business information.
            Output ONE of the following:

            - unreachable → Site does not load at all (HTTP errors, DNS errors, SSL issues, or redirects to unrelated sites).
            - no_info → Site loads but has no meaningful business content (e.g., “coming soon,” placeholder, or blank page).
            - little_info → Site has some business content, but insufficient to understand offerings fully.
                Examples: Only a homepage with vague text, only contact info, or minimal services description.
            - much_info → Full website with detailed sections: services/products, About Us, team, case studies, testimonials, contact info.
                - Website should clearly describe the business operations.

            Evaluation tips:
                - Check homepage + main pages (About, Services, Contact).
                - Consider clarity, depth, and relevance of information.
                - Include business info even if partially visible or scattered.

        7. "confidence": Provide a number between 0 and 100 representing confidence in your classification.

            Guidelines for scoring:

            - 90-100 → Very confident:
                - Verified info from official website, LinkedIn, Crunchbase, business directories, news articles.
                - Website fully functional with detailed info.
            - 70-89 → Moderate confidence:
                - Partial evidence from smaller websites, social media, or limited listings.
                - Website functional but missing some info.
            - 40-69 → Low confidence:
                - Weak or ambiguous online presence.
                - Minimal website or inconsistent info across sources.
            - 0-39 → Very low confidence:
                - No reliable sources.
                - Site down, parked, or major ambiguity.
                - Mostly guesswork.

            Additional rules:
                - Reduce confidence if website is minimal or inaccessible, even if other sources exist.
                - Increase confidence if multiple sources confirm the business type, size, and website info.

        Additional Rules:
        - Respond ONLY with valid JSON (no markdown, no explanations).
        - Always include all keys, even if unsure (use "Unknown" or "N/A" if needed).
        - If the domain is parked, redirected, or inaccessible, reflect this in "website" and adjust "confidence" accordingly.
        - Cross-validate information from multiple sources whenever possible.
        - Output must start with '{{' and end with '}}'.

        Example output:
        {{
            "domain": "{domain}",
            "category": "Technology and software development",
            "other_industry_category": "N/A",
            "company_size": "20 to 50",
            "email_provider": "No",
            "website": "much_info",
            "confidence": 80
        }}

        Domain to analyze: {domain}
        """

        try:
            req_start = time.time()
            res = call_gemini(prompt)

            try:
                js = json.loads(res)
            except Exception:
                print(f"⚠ Invalid JSON from Gemini for {domain}. Storing Unknowns.")
                js = {}

            req_time = round(time.time() - req_start, 2)
            print(f"📝 Gemini output for {domain}:")
            print(json.dumps(js, indent=2))
            print(f"⏱ Response time: {req_time} seconds")

            # --- Assign outputs ---
            df.at[idx, 'category'] = js.get('category', 'Unknown')
            df.at[idx, 'company_size'] = js.get('company_size', 'Unknown')
            df.at[idx, 'email_provider'] = js.get('email_provider', 'Unknown')
            df.at[idx, 'confidence'] = int(js.get('confidence', 0))
            df.at[idx, 'other_industry_category'] = js.get('other_industry_category', 'N/A')
            df.at[idx, 'website'] = js.get('website', 'Unknown')

        except Exception as e:
            print(f"⚠ Error at {domain}: {e}")
            df.at[idx, 'category'] = "Unknown"
            df.at[idx, 'company_size'] = "Unknown"
            df.at[idx, 'email_provider'] = "Unknown"
            df.at[idx, 'confidence'] = 0
            df.at[idx, 'other_industry_category'] = "N/A"
            df.at[idx, 'website'] = "Unknown"

        # --- Save progress safely ---
        df.to_csv(save_path, index=False)
        processed += 1
        print(f"💾 Saved progress. Rows completed: {processed}")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving progress...")
    df.to_csv(save_path, index=False)

# ------------ Completion Summary ------------
end_time = datetime.now()
elapsed = end_time - start_time
df.to_csv(save_path, index=False)

print("\n✅ Enrichment complete!")
print(f"⏱ Start time: {start_time}")
print(f"⏱ End time: {end_time}")
print(f"⏱ Total time taken: {elapsed}")
print(f"📁 Saved to: {save_path}")
