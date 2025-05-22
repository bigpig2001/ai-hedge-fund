import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Need to ensure that src.llm can be found.
# This is a common pattern in testing when dealing with relative imports.
# We add the parent directory of 'src' to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.llm.gemini_api import generate_text_gemini
# Mock google.generativeai before importing the module that uses it
# This avoids genai.configure being called with a real key during import
mock_genai = MagicMock()

# Apply the patch to 'google.generativeai' in sys.modules
# This is effective if 'google.generativeai' has not been imported elsewhere directly yet.
# A more robust way for modules that configure on import is to structure them to allow late configuration
# or use environment variables that tests can control.
# For this case, we'll assume simple module-level patching works or adjust.

# If google.generativeai is already imported, this simple assignment might not be enough.
# A common pattern is to ensure the module under test is imported *after* patches are set up.
# We can use unittest.mock.patch.dict for os.environ and patch for google.generativeai.

# Let's try patching at the class level or within each test for clarity.

# Import the function to test *after* initial mocks if necessary, or manage module reload.
# For now, direct import and careful patching in tests.
from src.llm.gemini_api import generate_text_gemini 

# Mock response object structure from Gemini
class MockGeminiResponse:
    def __init__(self, text_content=None, error=None):
        if error:
            # This is for simulating errors during generation, not a property of the response object itself
            raise error 
        
        # Gemini's response.text directly gives the text if parts exist.
        # If response.parts is empty or generation failed, response.text might raise or be None.
        # The actual response object has a `parts` attribute which is a list of Part objects.
        # response.text is a helper that tries to join text from parts.
        # For simplicity, we'll mock response.text behavior.
        if text_content is not None:
            self.text = text_content
            self.parts = [MagicMock(text=text_content)] # Simulate having parts
        else:
            # Simulate a response with no text output (e.g. safety block)
            self.text = None # Or raise an error if that's what .text does
            self.parts = []


class TestGeminiApi(unittest.TestCase):

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True)
    @patch('src.llm.gemini_api.genai') # Patch genai within the scope of gemini_api.py
    def test_generate_text_gemini_success(self, mock_genai_module):
        # Configure the mock for genai module used in gemini_api.py
        mock_model_instance = MagicMock()
        mock_genai_module.GenerativeModel.return_value = mock_model_instance
        
        # Simulate a successful response with text
        mock_model_instance.generate_content.return_value = MockGeminiResponse(text_content="Gemini says hello!")

        prompt = "Hello Gemini!"
        response_text = generate_text_gemini(prompt)

        # Check that genai.configure was called (it's called at module import time)
        # This assertion depends on how the module is loaded and patched.
        # If gemini_api.py was imported before this patch, configure might have already run.
        # For robustness, it's better if configure is called by the function or class init.
        # Given the current structure of gemini_api.py, configure is called on import.
        # We are patching 'src.llm.gemini_api.genai', so configure should be on this mock.
        mock_genai_module.configure.assert_called_with(api_key="test_api_key")
        
        mock_genai_module.GenerativeModel.assert_called_once_with('gemini-pro')
        mock_model_instance.generate_content.assert_called_once_with(prompt)
        self.assertEqual(response_text, "Gemini says hello!")

    @patch.dict(os.environ, {}, clear=True) # Ensure GOOGLE_API_KEY is NOT set
    @patch('src.llm.gemini_api.genai') # Patch genai, though it might not be used if key is missing
    def test_generate_text_gemini_api_key_not_found(self, mock_genai_module):
        # Important: Need to reload the module under test for os.environ patch to affect its import-time logic
        # This is a more advanced testing pattern.
        # For now, we'll assume the current structure of gemini_api.py prints an error
        # and genai.configure might not be called or might be called with None.
        # The function generate_text_gemini itself checks for API key implicitly via genai state.

        # If API key is not found, genai.configure would have failed at module import.
        # The generate_text_gemini function itself doesn't re-check os.environ.
        # It relies on genai being configured.
        
        # Let's simulate genai.configure not being effective or model loading failing.
        # The original code has a try-except for genai.configure(api_key=GOOGLE_API_KEY)
        # If GOOGLE_API_KEY is missing, genai.configure is not called with a key.
        # The generate_text_gemini function will then likely fail when trying to use GenerativeModel.
        
        mock_genai_module.GenerativeModel.side_effect = ValueError("API key not configured")

        # To properly test the import-time failure, we'd need to reload src.llm.gemini_api
        # For this test, we'll assume the function proceeds and fails at model instantiation
        # if the API key wasn't set during the initial import.
        
        # If the module's import-time `genai.configure` failed due to missing key,
        # then `generate_text_gemini` attempting to use `genai.GenerativeModel` would fail.
        # The `generate_text_gemini` function has a try-except ValueError for this.
        
        response_text = generate_text_gemini("Any prompt")
        self.assertIsNone(response_text)
        # We expect it to print "Error: GOOGLE_API_KEY environment variable not set." at import time
        # and then "Error with Gemini API configuration or input: API key not configured" during call.


    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True)
    @patch('src.llm.gemini_api.genai')
    def test_generate_text_gemini_api_error(self, mock_genai_module):
        mock_model_instance = MagicMock()
        mock_genai_module.GenerativeModel.return_value = mock_model_instance
        
        # Simulate an API error during content generation
        mock_model_instance.generate_content.side_effect = Exception("Gemini server is down")

        response_text = generate_text_gemini("Any prompt")
        self.assertIsNone(response_text)
        mock_genai_module.configure.assert_called_with(api_key="test_api_key") # From import
        mock_model_instance.generate_content.assert_called_once()
        # We'd also expect "An unexpected error occurred while calling Gemini API: Gemini server is down" to be printed.

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True)
    @patch('src.llm.gemini_api.genai')
    def test_generate_text_gemini_malformed_response_no_text(self, mock_genai_module):
        mock_model_instance = MagicMock()
        mock_genai_module.GenerativeModel.return_value = mock_model_instance
        
        # Simulate a response that has no .text attribute or parts
        malformed_response = MockGeminiResponse(text_content=None) # No text, empty parts
        mock_model_instance.generate_content.return_value = malformed_response
        
        response_text = generate_text_gemini("Any prompt")
        self.assertIsNone(response_text)
        # Expect "Gemini API call did not return text..." to be printed.

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True)
    @patch('src.llm.gemini_api.genai')
    def test_generate_text_gemini_malformed_response_attribute_error(self, mock_genai_module):
        mock_model_instance = MagicMock()
        mock_genai_module.GenerativeModel.return_value = mock_model_instance
        
        # Simulate a response object that would cause an AttributeError if .text is accessed
        # For example, if response object itself is None or not what's expected.
        # The MockGeminiResponse handles this by setting .text to None if no text_content.
        # Let's try a more direct AttributeError by making .text raise it.
        class ResponseLackingText:
            def __init__(self):
                self.parts = [] # To satisfy "if response.parts"
            @property
            def text(self):
                raise AttributeError("Mock missing text property behavior")

        mock_model_instance.generate_content.return_value = ResponseLackingText()
        
        response_text = generate_text_gemini("Any prompt")
        
        # The code currently is:
        # if response.parts:
        #     return response.text
        # else:
        #     print("Gemini API call did not return text...")
        #     return None
        # An AttributeError on response.text would be caught by the outer `except Exception`.
        self.assertIsNone(response_text)
        # Expect "An unexpected error occurred..." to be printed.

if __name__ == '__main__':
    unittest.main()
