import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_connection():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
        return
    
    print("Testing Gemini API connection...")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Create a simple model instance with the correct model name
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Test with a simple prompt
        response = model.generate_content("Hello, are you working? Respond with 'Yes, I'm working!' if you can hear me.")
        
        # Print the response
        print("\nResponse from Gemini:")
        print(response.text)
        print("\n✅ Gemini API connection successful!")
        
    except Exception as e:
        print(f"\n❌ Error connecting to Gemini API: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your API key is correct")
        print("2. Check your internet connection")
        print("3. Make sure your API key has access to the Gemini API")
        print("4. Check if there are any API usage limits or quotas exceeded")

if __name__ == "__main__":
    test_gemini_connection()
