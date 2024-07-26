import pandas as pd
import re
import numpy as np
import requests
import asyncio

def wrong_category(data): #takes note of entries that have business models under category
  wrong_category = []
  business_model = ['OEM', 'End-user', 'System Integrator', 'Distributor']
  for i in range(len(data['Category'])):
      if data['Category'][i] in business_model:
        wrong_category.append(1)
      else:
        wrong_category.append(0)

  data["wrong_category"] = wrong_category

def wrong_subcategory(data): #takes note of entries that have business models under subcategory
  wrong_sub = []
  business_model = ['OEM', 'End-user', 'System Integrator', 'Distributor']
  for i in range(len(data['Subcategory'])):
      if data['Subcategory'][i] in business_model:
        wrong_sub.append(1)
      else:
        wrong_sub.append(0)

  data['wrong_subcategory'] = wrong_sub

def wrong_business_model(data): #updates format of standard business models and takes note of wrong business models
  business_model = ['OEM', 'End-user', 'System Integrator', 'Distributor']
  wrong = []

  for i in range(len(data)):
    if str(data["Business Model"][i]) not in business_model:
      x = str(data["Business Model"][i]).lower()

      if "services" in x:
        data.loc[i, 'Business Model'] = "System Integrator"

      elif "end" in x:
        data.loc[i, 'Business Model'] = "End-user"

      elif "oem" in x:
        data.loc[i, 'Business Model'] = "OEM"

      elif "distributor" in x:
        data.loc[i, 'Business Model'] = "Distributor"

      else:
        if "system" in x:
          data.loc[i, 'Business Model'] = "System Integrator"

  for i in range(len(data["Business Model"])):
    if str(data["Business Model"][i]) not in business_model:
      wrong.append(1)
    else:
      wrong.append(0)

  data['wrong business model'] = wrong

def same_address_for_companies(data): #takes note of companies with the same address
  address_counts = data.groupby('Address')['Company'].nunique().reset_index(name='Company_Count')

  data['Duplicate_Address'] = data['Address'].isin(address_counts[address_counts['Company_Count'] > 1]['Address'])

  mult_add = list(data[data['Duplicate_Address'] == True].index)

  mult = list(np.zeros(len(data)))

  for i in range(len(mult_add)):
    mult[mult_add[i]] = 1

  data.drop('Duplicate_Address', axis = 1, inplace=True)
  data["address in multiple companies"] = mult

def similar_addresses(data): #takes note of addresses under the same company that differ in format
  data['Address'] = df['Address'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
  company_address = data.groupby("Company")["Address"].nunique()
  company_address = company_address[company_address > 1]
  company_names = company_address.index

  space_col = list(np.zeros(len(data)))
  char_col = list(np.zeros(len(data)))
  letter_col = list(np.zeros(len(data)))
  all_col = list(np.zeros(len(data)))

  for y in range(len(company_names)):
    space = []
    extra_character = []
    letter_case = []
    all = []
    index = []
    sample = data[data['Company'] == company_names[y]]['Address'].duplicated(keep=False)
    sample_index = list(sample.index)

    for i in range(len(sample_index)):
      if sample[sample_index[i]] == False:
        index.append(sample_index[i])

    for i in range(len(index)):
      data_point = data[(data['Company'] == company_names[y])]['Address']
      space.append(str(data_point[index[i]]).replace(" ",""))
      extra_character.append(re.sub('[\\W\\d_]+', '', str(data_point[index[i]])))
      letter_case.append(str(data_point[index[i]]).lower())
      all.append(re.sub('[\\W\\d_]+', '', str(data_point[index[i]])).replace(" ","").lower())

    space2, char,letter,all2 = [],[],[],[]

    for i in range(len(index)):
      for j in range(len(index)):
        if i != j:
          if space[i] == space[j]:
            if index[i] not in space2:
              space2.append(index[i])

            if index[j] not in space2:
              space2.append(index[j])

          elif extra_character[i] == extra_character[j]:
            if index[i] not in char:
              char.append(index[i])

            if index[j] not in char:
              char.append(index[j])

          elif letter_case[i] == letter_case[j]:
            if index[i] not in letter:
              letter.append(index[i])

            if index[j] not in letter:
              letter.append(index[j])

          else:
            if all[i] == all[j]:
              if index[i] not in all2:
                all2.append(index[i])

              if index[j] not in all2:
                all2.append(index[j])

    for i in range(len(space2)):
      space_col[space2[i]] = 1
    for i in range(len(char)):
      char_col[char[i]] = 1
    for i in range(len(letter)):
      letter_col[letter[i]] = 1
    for i in range(len(all2)):
      all_col[all2[i]] = 1

  data["invalid address (extra space)"] = space_col
  data["invalid address (extra character)"] = char_col
  data["invalid address (letter case)"] = letter_col
  data["invalid address (extra space,character,letter case)"] = all_col

def wrong_contact_number(data): #takes note of contact numbers that contains strings/wrong number format
  def contains_letters(string):
    return bool(re.search(r'[a-zA-Z]', string))

  letter_no = []

  data.fillna({'Contact No.': ''}, inplace = True)

  for i in range(len(data)):
    if contains_letters(str(data["Contact No."][i])) == True:
      letter_no.append(1)
    else:
      letter_no.append(0)

  data["invalid contact no."] = letter_no

def wrong_fax_number(data): #takes note of fax numbers that contains strings/wrong number format
  def contains_letters(string):
    return bool(re.search(r'[a-zA-Z]', string))

  letter =[]

  data.fillna({'Fax No.': ''}, inplace = True)

  for i in range(len(data)):
    if contains_letters(str(data["Fax No."][i])) == True:
      letter.append(1)
    else:
      letter.append(0)

  data["invalid fax no."] = letter

def contains_only_digits(string):
    if isinstance(string, str):
        return bool(re.match(r'^\d+$', string))
    return False

def find_and_write_invalid_employees_indices(data): #takes note of entries with invalid employee numbers

    data['Employees'] = data['Employees'].astype(str).str.replace(',', '').str.split('.', expand=True)[0]
    non_numeric_indices = [index for index, value in data['Employees'].items() if not contains_only_digits(value)]
    data['invalid_employees'] = 0
    data.loc[non_numeric_indices, 'invalid_employees'] = 1
    numeric_indices = data['Employees'].str.isdigit()
    data.loc[numeric_indices & (data['invalid_employees'] == 0), 'invalid_employees'] = 0

def get_sales_outliers(data): #takes note of sales outliers
    data_copy = data.copy()
    data_copy['Annual Sales'] = pd.to_numeric(data_copy['Annual Sales'].astype(str).str.replace(',', ''), errors='coerce')

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    Q1 = data_copy['Annual Sales'].quantile(0.25)
    Q3 = data_copy['Annual Sales'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = data_copy.index[(data_copy['Annual Sales'] < lower_bound) | (data_copy['Annual Sales'] > upper_bound)]
    data['outlier_annual_sale'] = 0
    data.loc[outlier_indices, 'outlier_annual_sale'] = 1

    non_numeric_indices = data_copy[data_copy['Annual Sales'].isna()].index
    data.loc[non_numeric_indices, 'outlier_annual_sale'] = 1
    data.loc[non_numeric_indices, 'Annual Sales'] = np.nan

def detect_outliers_by_state(data): #takes note of state outliers
    data['State'] = data['State'].apply(lambda state: state.strip(" ./,") if isinstance(state, str) else state)
    data['State'] = data['State'].apply(lambda state: state.capitalize() if isinstance(state, str) else state)
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

    grouped = data.groupby('State')
    data['State'] = data['State'].str.strip().str.lower()

    def detect_outliers(values):
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (values < lower_bound) | (values > upper_bound)

    state_outliers = {}

    outlier_indices = []

    for state, group in grouped:
        group['outlier_latitude'] = detect_outliers(group['Latitude']).astype(int)
        group['outlier_longitude'] = detect_outliers(group['Longitude']).astype(int)

        group['outlier_latitude'] = (group['outlier_latitude'] == 1).astype(int)
        group['outlier_longitude'] = (group['outlier_longitude'] == 1).astype(int)

        state_outliers[state] = {'outlier_latitude': group['outlier_latitude'].tolist(),
                                 'outlier_longitude': group['outlier_longitude'].tolist()}

        data.loc[group.index, ['outlier_latitude', 'outlier_longitude']] = group[['outlier_latitude', 'outlier_longitude']].values

        outlier_indices.extend(group[group['outlier_latitude'] == 1].index)
        outlier_indices.extend(group[group['outlier_longitude'] == 1].index)

    data[['outlier_latitude', 'outlier_longitude']] = data[['outlier_latitude', 'outlier_longitude']].fillna(1)
    non_numeric_indices_long = data[data['Longitude'].isna()].index
    non_numeric_indices_lat = data[data['Latitude'].isna()].index
    data.loc[non_numeric_indices_long, 'outlier_longitude'] = 1
    data.loc[non_numeric_indices_lat, 'outlier_latitude'] = 1
    data['State'] = data['State'].str.title()

    return data, state_outliers

def clean_email_and_website(data): #interchanges wrong entries under website and email
    data.fillna({'Email': ''}, inplace = True)
    data.fillna({'Website': ''}, inplace = True)

    email_pattern = re.compile(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
    website_pattern = re.compile(r'\b(?:https?://)?(?:www\.)?([^\s]+)\b')

    emails = data['Email'].str.extract(email_pattern, expand=False)
    websites = data['Website'].str.extract(website_pattern, expand=False)

    email_contains_website_mask = emails.str.contains(r'www\.', na=False)
    website_contains_email_mask = websites.str.contains(r'@', na=False)

    data.loc[email_contains_website_mask, ['Email', 'Website']] = data.loc[email_contains_website_mask, ['Website', 'Email']].values
    data.loc[website_contains_email_mask, ['Email', 'Website']] = data.loc[website_contains_email_mask, ['Website', 'Email']].values

    empty_website_mask = (data['Website'] == '') & data['Email'].str.contains(r'http[s]?://|www\.', na=False)
    data.loc[empty_website_mask, ['Email', 'Website']] = data.loc[empty_website_mask, ['Website', 'Email']].values

    return data

def check_state_in_address(data):
    # Function to handle NaN and non-string values
    def clean_string(value):
        if isinstance(value, str):
            return value.lower()
        else:
            return str(value).lower()  # Convert non-string to string and then to lowercase
    
    # Apply cleaning function to State and Address columns
    data['State'] = data['State'].apply(clean_string)
    data['Address'] = data['Address'].apply(clean_string)
    
    # Create new column state_not_in_address based on the comparison
    data['state_not_in_address'] = data.apply(lambda row: 0 if row['State'] in row['Address'] else 1, axis=1)
    
    return data

def invalid_state(state, valid_states):
    if isinstance(state, str):  # Check if state is already a string
        state = state.strip()   # Strip leading/trailing whitespace if it's a string
        state = ''.join(state.split())
    if state not in valid_states:
        return 1
    else:
        return 0

file_name = str(input('Enter File Name (include .csv): '))
valid_states = input('Enter Valid State List (comma-separated): ').split(',')
valid_states = [' '.join(word.capitalize() for word in state.split()) for state in valid_states]
valid_states = [valid_state.replace(' ', '') for valid_state in valid_states]
df = pd.read_csv(file_name)
df = df.drop_duplicates(ignore_index = True) #drops duplicate entries
df['invalid_state'] = df['State'].apply(lambda x: invalid_state(x, valid_states))
check_state_in_address(df)
wrong_business_model(df)
wrong_category(df)
wrong_subcategory(df)
same_address_for_companies(df)
wrong_contact_number(df)
wrong_fax_number(df)
similar_addresses(df)
contains_only_digits(df['Employees'])
find_and_write_invalid_employees_indices(df)
get_sales_outliers(df)
detect_outliers_by_state(df)
clean_email_and_website(df)

# with open('API-Key.txt', 'r') as file: #reads the API key
#   GOOGLE_MAPS_API_KEY = str(file.read())

# async def get_lat_and_lng(address): #collect the coordinates from google maps 
#     try :
#         url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_MAPS_API_KEY}"
#         response = await asyncio.get_event_loop().run_in_executor(
#             None, lambda: requests.get(url)
#         )
#         data = response.json()
#         lat = data["results"][0]["geometry"]["location"]["lat"]
#         lng = data["results"][0]["geometry"]["location"]["lng"]

#         return {"latitude": lat, "longitude": lng}
#     except Exception as e:
#         print("################### [ Error in getting coordinates ] ######################")
#         print(f"Error : {e}")
#         return {"latitude": None, "longitude": None}

# async def main():

#   addresses = df["Address"]
#   coordinates = []

#   for i in range(len(addresses)):
#       if df['outlier_latitude'][i] == 1 or df['outlier_longitude'][i] == 1: #gets coordinates of entries with no coordinates or marked as an outlier
#         print(f"Getting coordinates for address {i + 1}")
#         print(addresses[i])

#         address = addresses[i]
#         coordinates.append(
#             {"address": address, "coordinates": await get_lat_and_lng(address)}
#         )

#         print(f"Latitude: {coordinates[-1]['coordinates']['latitude']}")
#         print(f"Longitude: {coordinates[-1]['coordinates']['longitude']}\n")

#         df.loc[i, 'Latitude'] = coordinates[-1]['coordinates']['latitude']
#         df.loc[i, 'Longitude'] = coordinates[-1]['coordinates']['longitude'] 

# asyncio.run(main())

detect_outliers_by_state(df) # updates the entries marked as missing coordinates but already have coordinates


save_name = str(input('Enter Save Name (include .csv): '))
df.to_csv(save_name, index = False)
