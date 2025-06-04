import requests

API_KEY = 'yMa_3I18llD30VYtU-BrOg'  # Get this from Proxycurl dashboard

def get_linkedin_profile_data(linkedin_url):
    endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    params = {
        "url": linkedin_url,
        "use_cache": "if-present"  # Faster + cheaper
    }

    response = requests.get(endpoint, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            "full_name": data.get("full_name"),
            "headline": data.get("headline"),
            "location": data.get("location"),
            "country": data.get("country_full_name"),
            "current_title": data.get("experiences", [{}])[0].get("title"),
            "current_company": data.get("experiences", [{}])[0].get("company"),
            "previous_experience": [exp.get("title") for exp in data.get("experiences", [])[1:3]],
            "skills": data.get("skills"),
            "industry": data.get("industry"),
            "education": [edu.get("school") for edu in data.get("education", [])]
        }
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    linkedin_url = "https://www.linkedin.com/in/vgulla/"
    profile_info = get_linkedin_profile_data(linkedin_url)
    print(profile_info)
