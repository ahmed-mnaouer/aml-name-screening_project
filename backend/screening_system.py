# AML Watchlist Screening System

# Step 1: Libraries and Data Loading
print("=== STEP 1: DATA LOADING ===")
print()

# Libraries
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity, levenshtein_distance, soundex
from googletrans import Translator
import re
import warnings
warnings.filterwarnings('ignore')



# Loading the dataset
df = pd.read_excel(r"C:\Users\ahmed\OneDrive\Desktop\aml-name-screening_project-main\dataset\cleaned_aml_data.xlsx")

# Make sure it is loaded
print(f"Dataset loaded successfully!")
print(f"Original dataset shape: {df.shape}")
print(f"Columns in dataset: {list(df.columns)}")

# We will keep all columns for a more informative display later on about the matching customer from the dataset

# Remove rows with missing names or risk categories
df = df.dropna(subset=['Full Name', 'Risk Category'])

# View it again in case some rows got removed during the last step
print(f"After cleaning: {df.shape}")
print(f"Unique risk categories: {df['Risk Category'].unique()}")

# Rename 'Age / DOB' column to 'Age'
if 'Age / DOB' in df.columns:
    df.rename(columns={'Age / DOB': 'Age'}, inplace=True)

# Convert Age column from dates to numbers if it exists
if 'Age' in df.columns:
    print("Processing Age column...")
    from datetime import datetime
    
    for i, age_value in enumerate(df['Age']):
        if pd.notna(age_value):
            age_str = str(age_value).strip()
            
            # Check if it's already a number (like "50")
            if age_str.isdigit():
                df.iloc[i, df.columns.get_loc('Age')] = int(age_str)
            
            # Check if it's a date format (like "1982-07-14")
            elif len(age_str) == 10 and '-' in age_str:
                try:
                    # Parse the date and calculate age
                    birth_date = datetime.strptime(age_str, '%Y-%m-%d')
                    current_year = datetime.now().year
                    age = current_year - birth_date.year
                    df.iloc[i, df.columns.get_loc('Age')] = age
                except:
                    # If parsing fails, keep original value
                    pass

# Initializing translator for Arabic to English/French translation
translator = Translator()

# Translate Arabic names to English (if any)
import asyncio
async def translate_arabic_names(df):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    tasks = []
    indices = []
    for i, name in enumerate(df['Full Name']):
        if pd.notna(name) and arabic_pattern.search(str(name)):
            tasks.append(translator.translate(str(name), src='ar', dest='en'))
            indices.append(i)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, res in zip(indices, results):
        if hasattr(res, 'text') and res.text and res.text != df.iloc[idx, df.columns.get_loc('Full Name')]:
            # Remove any space before "-" after translation
            cleaned_text = re.sub(r'\s+-', '-', res.text)
            print(f"Original: {df.iloc[idx, df.columns.get_loc('Full Name')]} -> Translated: {cleaned_text}")
            df.iloc[idx, df.columns.get_loc('Full Name')] = cleaned_text
        elif isinstance(res, Exception):
            print(f"Translation error at row {idx}: {res}")

asyncio.run(translate_arabic_names(df))

# Final Dataset Checking
print("Name preprocessing completed!")
print(f"Final dataset shape: {df.shape}")
print("\nSample of processed data:")
print(df.head())
print(f"\nRisk category distribution:")
print(df['Risk Category'].value_counts())
print()

# Main screening loop
while True:
    # Step 2: Similarity Calculation
    print("=== STEP 2: SIMILARITY CALCULATION ===")
    print()

    # Ask User to input the name to search for
    input_name = input("Enter Full Name : ")

    print(f"Searching for: '{input_name}'")

    # Find best match by checking all names in dataset
    best_match_name = ""
    best_match_risk = ""
    best_match_nationality = ""
    best_score = 0

    for index, row in df.iterrows():
        dataset_name = row['Full Name']
        
        # Convert names to lowercase for comparison
        name1 = input_name.lower().strip()
        name2 = dataset_name.lower().strip()
        
        # 1. Jaro-Winkler similarity (50% weight)
        jaro_winkler = jaro_winkler_similarity(name1, name2) * 100
        
        # 2. Levenshtein similarity (35% weight)
        # Convert distance to similarity (0-100 scale)
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            levenshtein_sim = 100
        else:
            lev_distance = levenshtein_distance(name1, name2)
            levenshtein_sim = (1 - lev_distance / max_len) * 100
        
        # 3. Soundex similarity - binary Soundex similarity (0 or 100) (15% weight)
        soundex1 = soundex(name1)
        soundex2 = soundex(name2)
        soundex_sim = 100 if soundex1 == soundex2 else 0
        
        # Calculate weighted average
        final_score = (jaro_winkler * 0.5) + (levenshtein_sim * 0.35) + (soundex_sim * 0.15)
        final_score = round(final_score, 2)

        # Keep the best match
        if final_score > best_score:
            best_score = final_score
            best_match_name = dataset_name
            best_match_risk = row['Risk Category']
            if 'Nationality' in row:
                best_match_nationality = row['Nationality']

    # Display results
    print(f"Best match found: '{best_match_name}'")
    print(f"Similarity score: {best_score}%")
    print(f"Risk category: {best_match_risk}")
    if best_match_nationality:
        print(f"Nationality: {best_match_nationality}")

    print()
    print("=== SIMILARITY CALCULATION COMPLETE ===")
    print()

    print("=== STEP 3: DECISION MAKING ===")
    print()

    # Determine if customer is high risk or medium risk
    if best_match_risk in ["Terrorism Financing", "Money Laundering"]:
        risk_level = "High Risk"
    elif best_match_risk == "PEP":
        risk_level = "Medium Risk"
    else:
        risk_level = "Unknown Risk"


    print(f"Similarity Score: {best_score}%")
    print(f"Corresponding Risk Level of matching customer: {risk_level}")
    print()

    # Apply decision rules
    if best_score >= 75 and risk_level in ["High Risk", "Medium Risk"]:
        decision = "BLOCKED"
        reason = "High similarity match with high/medium risk customer"
        
    elif 30 <= best_score < 75 and risk_level == "High Risk":
        decision = "BLOCKED"
        reason = "High-risk match found"
        
    elif 30 <= best_score < 75 and risk_level == "Medium Risk":
        decision = "AMBIGUOUS"
        reason = "Medium-risk match found - manual review required"
        
    elif best_score < 30:
        decision = "ALLOWED"
        reason = "No major matches found"
        
    else:
        decision = "ALLOWED"
        reason = "No significant risk identified"

    # Final decision
    print("=== FINAL DECISION ===")
    print()
    print(f"DECISION: {decision}")
    print(f"REASON: {reason}")
    print(f"MATCH: '{best_match_name}' ({best_score}% similarity)")
    print(f"RISK: {best_match_risk} ({risk_level})")
    print()
    print("=== AML SCREENING COMPLETE ===")
    print()
    
    # Ask user if they want to screen another name
    while True:
        continue_screening = input("Do you want to screen another name? (y/n): ").lower().strip()
        if continue_screening in ['y', 'yes']:
            print("\n" + "="*50 + "\n")
            break
        elif continue_screening in ['n', 'no']:
            print("Thank you for using the AML Watchlist Screening System!")
            exit()
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")