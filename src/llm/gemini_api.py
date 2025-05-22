import os
import google.generativeai as genai

# Configure API Key
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    # Potentially raise an exception here or handle it as appropriate for the application
    # For now, we'll let it proceed, but API calls will fail.
    pass
except Exception as e:
    print(f"An error occurred during Gemini API configuration: {e}")
    pass

def generate_text_gemini(prompt: str) -> str | None:
    """
    Generates text using the Gemini API.

    Args:
        prompt: The prompt to send to the Gemini API.

    Returns:
        The generated text response, or None if an error occurs.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Ensure response has text parts and access the first one
        if response.parts:
            return response.text
        else:
            # This case handles scenarios where the response might be blocked
            # or doesn't contain text, which can happen with safety settings or empty prompts.
            print("Gemini API call did not return text. This might be due to safety settings or an empty response.")
            # Log the full response for debugging if possible and appropriate
            # print(f"Full Gemini response: {response}") 
            return None
            
    except ValueError as ve:
        # This can happen if the API key is not configured/valid
        print(f"Error with Gemini API configuration or input: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while calling Gemini API: {e}")
        return None

if __name__ == '__main__':
    # Example usage (requires GOOGLE_API_KEY to be set in the environment)
    if 'GOOGLE_API_KEY' in os.environ: # Check if API key was loaded
        test_prompt = "Explain the theory of relativity in simple terms."
        print(f"Sending prompt to Gemini: \"{test_prompt}\"")
        generated_text = generate_text_gemini(test_prompt)
        if generated_text:
            print(f"\nGemini Response:\n{generated_text}")
        else:
            print("\nFailed to get a response from Gemini.")
    else:
        print("Cannot run example: GOOGLE_API_KEY is not configured.")
