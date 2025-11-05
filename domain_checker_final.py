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
    os.getenv("GEMINI_API_KEY_3")
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
file_path = r'C:/SAG/Air-Channel/files(Oct21-21)/converted_users.csv'
save_path = r'C:/SAG/Air-Channel/files(Oct21-21)/converted_users_enriched.csv'

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
        You are an autonomous enrichment and classification agent. Your goal is to investigate the company behind the domain provided and return a single JSON object with accurate classifications, size estimates, and website evaluation.

        Follow all instructions carefully.

        Your workflow:

            A. First, visit the website {domain}.
            - Attempt to load the site and explore all available sections.
            - Extract all visible text and meaningful content (About, Services, Products, Contact, etc.).
            - If the website provides sufficient information to confidently determine the company’s category, size, and website evaluation, STOP here and use this data for classification.

            B. If the website provides little or no usable information (e.g., unreachable, parked, minimal, or generic content), THEN search the domain name on external platforms including Google, Bing, and public company directories.
            - Review the top 10 search results.
            - Collect and cross-check information about the company, ensuring the results refer to the same organization.
            - Use this additional information to fill in missing details and improve confidence.

            C. Combine insights from both the website (if available) and external sources (if needed) before making your final decision.

        You must return a single JSON object following the exact structure and rules below.

        ---

        Guidelines:

        1. "domain": Return the exact domain given as input.

        2. "category": Choose ONE of the following EXACT options:
            - Technology and Software Development
            - Professional Services
            - Marketing and Advertising
            - E-commerce and Retail
            - Financial Service
            - Education and e-learning
            - Real Estate and Property Management
            - Healthcare
            - Logistics and Transportation
            - Manufacturing and Industrial
            - Other

        3. "other_industry_category":
            - Include this ONLY if "category" = "Other".
            - Describe the most relevant industry type (e.g., "Legal Services", "Travel Agency", "Non-profit", "Entertainment").
            - If "category" is not "Other", set this to "N/A".

        4. "company_size": Estimate the number of employees only if reliable clues are available.
        Choose ONE of these options:
            - solo
            - 1 to 5
            - 5 to 20
            - 20 to 50
            - 50 to 100
            - 100 to 200
            - 200 to 500
            - 500+
        Estimation rules:
            - Use explicit info from the website, social media, or verified external profiles (e.g., “team of 8”, “over 200 employees”).
            - If no such data exists, set this to "Unknown".
            - Do NOT guess based only on design quality or website scale.

        5. "email_provider": Identify if the domain is primarily used as an email service.
            - Output "Yes" if it is an email provider.
            - Output "No" if it is a regular business or organization.
            Examples:
                - gmail.com → Yes
                - yahoo.com → Yes
                - company.com → No

        6. "website": Evaluate the domain's website for usable business information.
            Output ONE of the following:
                - unreachable → Site does not load at all.
                - no_info → Site loads but has no meaningful business content.
                - little_info → Some content but insufficient to clearly understand the business.
                - much_info → Detailed site with clear business information.

        7. "confidence": Provide a number between 0 and 100 representing confidence in the final classification.
            Guidelines:
                - 90-100 → Very confident (clear and consistent info from site and/or sources).
                - 70-89 → Moderate confidence (some data but partial).
                - 40-69 → Low confidence (limited presence or unclear info).
                - 0-39 → Very low confidence (no reliable info).

        Additional Rules:
        - Respond ONLY with valid JSON. No markdown or explanations.
        - Always include all keys even if uncertain ("Unknown" or "N/A" when needed).
        - If the domain is parked or unreachable, reflect that in "website" and lower confidence.
        - Prefer direct website data. Use external sources only when website data is missing or insufficient.
        - Output must start with '{{' and end with '}}'.

        Example output:
        {{
            "domain": "{domain}",
            "category": "Technology and Software Development",
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
