import requests
import json

def test_api():
    url = "https://api.xiaomimimo.com/v1/chat/completions"

    headers = {
        "api-key": "sk-skcth7q2ul7tuz1w10hfdooedrob3a86b29pvtttj9l5vd45",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mimo-v1-7b-rl",
        "messages": [
            {
                "role": "system",
                "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example-files.cnbj1.mi-fds.com/example-files/image/image_example.png"
                        }
                    },
                    {
                        "type": "text",
                        "text": "please describe the content of the image"
                    }
                ]
            }
        ],
        "max_completion_tokens": 1024
    }

    print(f"Sending request to {url}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        print("Response Body:")
        
        try:
            # Parse JSON nicely if possible
            parsed_json = response.json()
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError:
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_api()
