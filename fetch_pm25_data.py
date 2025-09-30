import requests
import pandas as pd
import json

AQS_USER_EMAIL = "edwardalkl@outlook.com"
AQS_PW_KEY = "mauvebird78" 

# Define the base API endpoint for hourly data by county
API_URL = "https://aqs.epa.gov/data/api/sampledata/bycounty"

# Define the query parameters for LA County PM2.5 data
# NOTE: The API limits requests to one year or less. You may need to loop or run this multiple times.
params = {
    "email": AQS_USER_EMAIL,
    "key": AQS_PW_KEY,
    "param": "44201",          # Parameter Code for PM2.5 (FRM/FEM)
    "bdate": "20240101",       # Start Date (YYYYMMDD format)
    "edate": "20240229",       # End Date
    "state": "06",             # State FIPS Code for California
    "county": "037",           # County FIPS Code for Los Angeles County
}

# --- 2. SEND API REQUEST AND HANDLE RESPONSE ---
print(f"Sending API request for PM2.5 data from {params['bdate']} to {params['edate']}...")

try:
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    data = response.json()

    # --- 3. PROCESS JSON DATA ---
    if data.get('Header') and data['Header']['status'] != 'Success':
        print(f"API Request failed with status: {data['Header']['status']}")
        if data.get('Header', {}).get('error'):
            print(f"Error Message: {data['Header']['error']}")
        # Common errors: invalid key, date range too large, or no data found
        
    elif data and 'Data' in data and data['Data']:
        df = pd.DataFrame(data['Data'])
        
        # Select and rename key columns for clarity
        df = df.rename(columns={
            'date_local': 'Date',
            'time_local': 'Time',
            'sample_measurement': 'PM25_Concentration',
            'site_code': 'Site_ID',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
        })

        # Combine Date and Time into a single datetime column
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        
        # Filter for only the necessary columns
        final_df = df[['DateTime', 'Site_ID', 'Latitude', 'Longitude', 'PM25_Concentration']].copy()
        
        # Save the raw data
        output_filename = "la_pm25_raw_2024_q1.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"\n✅ Successfully retrieved and saved {len(final_df)} records to {output_filename}")
        
    else:
        print("API response contains no 'Data' records. Check date range or API usage limits.")

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred during the request: {e}")
    
except json.JSONDecodeError:
    print("\n❌ Failed to decode JSON response. Check the raw response content.")
    print(response.text[:500]) # Print first 500 chars for debugging
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")