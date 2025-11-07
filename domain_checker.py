import pandas as pd
import json
import itertools
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# ------------ Load API Keys ------------
load_dotenv()
api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5")
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
file_path = r'files/testing_domain.csv'
save_path = r'files/testing_enriched.csv'

df = pd.read_csv(file_path)

# Ensure required columns exist
for col in ['category','company_size','email_provider','confidence','other_industry_category','website','website_status','website_reason']:
    if col not in df.columns:
        df[col] = ""

# Ensure required columns exist and have correct dtype
text_columns = [
    'category', 'company_size', 'email_provider', 
    'other_industry_category', 'website', 
    'website_status', 'website_reason'
]
numeric_columns = ['confidence']

for col in text_columns:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str)

for col in numeric_columns:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

print(df.dtypes)


# ------------ Website Checking Functions ------------
def clean_html(html):
    """Remove HTML tags and scripts for plain readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)

def check_website(domain):
    """
    Check website reachability and return:
    - status: 'reachable' or 'unreachable'
    - reason: HTTP status code or error reason
    - text snippet (cleaned, max 4000 chars)
    """
    urls_to_try = [
        f"https://{domain}",
        f"http://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}"
    ]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=10, headers=headers)
            if r.status_code < 400 and "text/html" in r.headers.get("Content-Type", ""):
                return "reachable", str(r.status_code), clean_html(r.text)[:4000]
            else:
                return "unreachable", str(r.status_code), ""
        except requests.exceptions.Timeout:
            reason = "Timeout"
        except requests.exceptions.SSLError:
            reason = "SSL Error"
        except requests.exceptions.ConnectionError:
            reason = "Connection Error"
        except requests.RequestException as e:
            reason = str(e)
        continue

    return "unreachable", reason, ""

# ------------ Gemini API Call Function ------------
def call_gemini(prompt, retries=3):
    global current_key
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content(prompt)
            text = (response.text or "").strip()

            if not text:
                raise ValueError("Empty response from Gemini.")

            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                else:
                    text = text.replace("```json", "").replace("```", "").strip()

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
                time.sleep(60)
                continue
            elif "403" in err or "SERVICE_DISABLED" in err:
                print(f"🚫 API disabled for key {current_key[:6]}... Skipping key.")
                current_key = next(key_cycle)
                configure_genai(current_key)
                continue
            elif "Empty response" in err:
                time.sleep(20)
                continue
            else:
                raise RuntimeError(f"Gemini API error: {err}")
    raise RuntimeError("❌ All retries failed for Gemini API call.")

# ------------ Main Processing Loop ------------
start_time = datetime.now()
processed = 0

try:
    for idx, row in df.iterrows():
        category_value = str(row.get('category', '')).strip()
        # Skip rows that already have a meaningful category
        if category_value not in ["", "Unknown", "nan", "NaN"]:
            continue

        domain = row.get('domain')
        if not domain or pd.isna(domain):
            continue

        print(f"\n🌐 Processing: {domain}")

        # --- Check website ---
        site_status, site_reason, site_text = check_website(domain)
        print(f"🌍 Website status for {domain}: {site_status}, Reason: {site_reason}")

        # --- Updated Prompt with Real Site Info ---
        prompt = f"""
        You are an autonomous enrichment and classification agent. Your goal is to investigate the company behind the domain provided and return a single JSON object with accurate classifications, size estimates, confidence finding, and website evaluation.

        Follow all instructions carefully.

        A. Website Analysis:
        - Visit the website {domain}.
        - Load the site and explore sections like About, Services, Products, Contact.
        - Extract visible text and meaningful content.
        - If sufficient info exists, classify and stop. Otherwise, proceed to external verification.

        B. External Verification:
        - If website info is insufficient (unreachable, parked, minimal, or generic), search the domain on Google, Bing, and public directories.
        - Review the top 10 results and cross-check information.
        - Use this to fill missing details and improve confidence.

        C. Python-verified input data:
        - Website reachability: {site_status}
        - Extracted text snippet:
        \"\"\"{site_text}\"\"\"
        - Use site text if reachable; otherwise rely on external knowledge.

        D. Output Guidelines:

        1. "domain": exact input domain.

        2. "category": Choose ONE: Technology and Software Development, Professional Services, Marketing and Advertising, E-commerce and Retail, 
            Financial Service, Education and e-learning, Real Estate and Property Management, Healthcare, Logistics and Transportation,
            Manufacturing and Industrial, Other

        3. "other_industry_category": Only if "category" = Other, else "N/A". 
            Describe the most relevant category (e.g., "Legal Services", "Travel Agency", "Non-profit", "Entertainment").

        4. "company_size": solo, 1 to 5, 5 to 20, 20 to 50, 50 to 100, 100 to 200, 200 to 500, and 500+.
            Estimation rules:
            - Estimate company size using clues such as team description, About page, customer base, LinkedIn presence, or general business type.
            - If explicit numbers are missing, infer the **most likely range** based on wording (e.g., "our team", "global offices", "startup", "enterprise").
            - Never output "Unknown" — always choose the **closest possible range** from: solo, 1 to 5, 5 to 20, 20 to 50, 50 to 100, 100 to 200, 200 to 500, 500+.
            - Use "solo" if it appears to be an individual or freelancer website.

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
        - Output valid JSON only, no markdown or explanations.
          Always provide a valid value for every field — never use "Unknown" for company_size.
        - If company size is not explicitly mentioned, make a **best approximate guess** from available text or context.
        - Always include all keys, use "Unknown" or "N/A" when necessary.
        - Use {site_status} when deciding "website".
        - Prefer website data; use external sources only if necessary.
        - Output must start with '{{' and end with '}}'.

        Example:
        {{
        "domain": "{domain}",
        "category": "Technology and Software Development",
        "other_industry_category": "N/A",
        "company_size": "20 to 50",
        "email_provider": "No",
        "website": "much_info",
        "confidence": 85
        }}

        Domain to analyze: {domain}
        """

        # --- Call Gemini ---
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
            df.at[idx, 'website_status'] = site_status
            df.at[idx, 'website_reason'] = site_reason

        except Exception as e:
            print(f"⚠ Error at {domain}: {e}")
            df.at[idx, 'category'] = "Unknown"
            df.at[idx, 'company_size'] = "Unknown"
            df.at[idx, 'email_provider'] = "Unknown"
            df.at[idx, 'confidence'] = 0
            df.at[idx, 'other_industry_category'] = "N/A"
            df.at[idx, 'website'] = "Unknown"
            df.at[idx, 'website_status'] = site_status
            df.at[idx, 'website_reason'] = site_reason

        df.to_csv(save_path, index=False)
        processed += 1
        print(f"💾 Saved progress. Rows completed: {processed}")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Interrupted. Saving progress...")
    df.to_csv(save_path, index=False)

end_time = datetime.now()
elapsed = end_time - start_time
df.to_csv(save_path, index=False)
print("\n✅ Enrichment complete!")
print(f"⏱ Start: {start_time}, End: {end_time}, Total: {elapsed}")
print(f"📁 Saved to: {save_path}")