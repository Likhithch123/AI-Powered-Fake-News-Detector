import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = "your_google_api_key"
FACT_CHECK_API_KEY = "your-fact-check-api-key"

# Initialize LangChain Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash',google_api_key=GOOGLE_API_KEY)

# Function to get fact_checking data
def get_fact_check_results(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'claims' in data:
            return data['claims']
    return None


# Function to Calculate Confidence Score
def calculate_confidence_score(fact_checks, ai_analysis):
    score = 50  # Default Neutral Score

    if fact_checks:
        for claim in fact_checks:
            if "textualRating" in claim["claimReview"][0]:
                rating = claim["claimReview"][0]["textualRating"].lower()

                if "true" in rating:
                    score += 30  # Strong support
                elif "mostly true" in rating:
                    score += 20
                elif "misleading" in rating:
                    score -= 20
                elif "false" in rating:
                    score -= 30  # Strongly unreliable

    # AI Analysis Contribution
    if "misinformation" in ai_analysis.lower():
        score -= 15
    if "reliable" in ai_analysis.lower():
        score += 15

    # Ensure score is within 0-100 range
    return max(0, min(100, score))


# Streamlit UI
st.set_page_config(page_title='Fake News Detection')
st.title('📰AI-Powered Fake News & Content Detector')

st.write('An AI-powered tool that detects fake news articles using AI and NLP.')

st.write('Enter a news article or claim to analyze its authenticity.')

# User input
user_input = st.text_area("Paste the news article or claim here:")

if st.button('Analyze'):
    if user_input:
        messages = [
            SystemMessage(content="Analyze the credibility of this text and check if it's misinformation."),
            HumanMessage(content=user_input)
        ]
        response = llm(messages=messages)
        ai_analysis = response.content

        # Fact-Checking API
        fact_check_results = get_fact_check_results(user_input)

        # Confidence Score Calculation
        confidence_score = calculate_confidence_score(fact_check_results, ai_analysis)

        # Display Results
        st.subheader("🧠 AI Analysis:")
        st.write(ai_analysis)

        if fact_check_results:
            st.subheader("🔍Fact-Checking Sources:")
            for claim in fact_check_results:
                st.write(f'✅ The claim was fact-checked by **{claim['claimReview'][0]['publisher']['name']}**')
                st.write(f'**Claim:** {claim['text']}')
                st.write(f'🔗**Source:** {claim['claimReview'][0]['url']}')
                st.write(f'🏷**Conclusion:** {claim['claimReview'][0]['textualRating']}')
                st.write('-----')
        else:
            st.write('No matching fact-checking results found.')

        
        # Display Confidence Score
        st.subheader(f"🔹 Confidence Score: {confidence_score}%")
        if confidence_score >= 80:
            st.success("✅ This claim is highly reliable!")
        elif confidence_score >= 50:
            st.info("⚠️ This claim is unverified. Be cautious!")
        else:
            st.error("❌ This claim is likely false or misleading!")

    else:
        st.warning('Please enter a news article or claim.')