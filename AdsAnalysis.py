import streamlit as st
import pandas as pd
import nltk
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import io
import contextlib

# Function to suppress output
def suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())

# Ensure necessary NLTK data is downloaded
with suppress_stdout():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

# Define functions for additional readability metrics
def new_dale_chall(text):
    return textstat.dale_chall_readability_score(text)

def smog_index(text):
    return textstat.smog_index(text)

# Add other readability metrics as needed

st.title('CSV File Analysis')

# Upload CSV file
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

# Text input for single text analysis
text_input = st.text_area('Or paste text here for analysis')

if uploaded_file is not None or text_input:
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Show preview of the CSV file
        st.write('Preview of the CSV file:')
        st.write(df.head())
        
        # Select column to process
        column = st.selectbox('Select column to process', df.columns)
        
        # Extract text data from the selected column
        text_data_list = df[column].astype(str).tolist()
    else:
        # Use the pasted text for analysis
        text_data_list = [text_input]
    
    # Updated lists of sales-y, news-y, and CTA words
    salesy_words = ['buy', 'discount', 'offer', 'sale', 'save', 'deal', 'limited time', 'promotion', 'introductory', 
                    'exclusive', 'special', 'bonus', 'free', 'trial', 'demo', 'guarantee', 'risk-free', 'proven', 
                    'results', 'effective', 'trusted', 'leading', 'top', 'best', 'affordable', 'value', 'investment']
    
    newsy_words = ['news', 'report', 'update', 'announcement', 'release', 'alert', 'bulletin', 'briefing', 'dispatch', 
                   'headline', 'story', 'article', 'blog', 'post', 'journal', 'publication', 'review', 'white paper', 
                   'case study', 'webinar', 'conference', 'symposium', 'workshop', 'research', 'study', 'findings', 
                   'data', 'results', 'clinical trial', 'phase', 'FDA', 'approval', 'submission', 'grant', 'award', 
                   'funding', 'partnership', 'collaboration', 'acquisition', 'merger', 'launch', 'milestone', 
                   'breakthrough', 'innovation', 'discovery']
    
    cta_words = ['buy now', 'shop now', 'learn more', 'contact us', 'request a quote', 'get started', 'sign up', 
                 'register', 'download', 'subscribe', 'follow us', 'join us', 'visit our website', 'request a demo', 
                 'schedule a consultation', 'speak with an expert', 'explore our solutions', 'discover our technology', 
                 'learn about our research', 'partner with us', 'invest in the future', 'improve patient outcomes', 
                 'advance scientific discovery']

    # Generate Report button
    if st.button('Generate Report'):
        # Initialize Sentiment Analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Create a new DataFrame to store the results
        results = pd.DataFrame(columns=[
            'Text', 'Flesch-Kincaid Score', 'New Dale-Chall', 'SMOG Index', 'Lexical Diversity', 'Top Words', 
            'Neg Sentiment', 'Neu Sentiment', 'Pos Sentiment', 'Compound Sentiment', 'Final Sentiment',
            'Top CTA Words', 'Sales-y Words Count', 'News-y Words Count'
        ])
        
        # Initialize totals for averaging
        total_flesch_kincaid = 0
        total_dale_chall = 0
        total_smog = 0
        total_lexical_diversity = 0
        total_neg_sentiment = 0
        total_neu_sentiment = 0
        total_pos_sentiment = 0
        total_compound_sentiment = 0
        total_salesy_count = 0
        total_newsy_count = 0
        
        # Process each text data
        for text_data in text_data_list:
            # Flesch-Kincaid score
            flesch_kincaid_score = textstat.flesch_kincaid_grade(text_data)
            total_flesch_kincaid += flesch_kincaid_score
            
            # New Dale-Chall score
            dale_chall_score = new_dale_chall(text_data)
            total_dale_chall += dale_chall_score
            
            # SMOG Index
            smog_score = smog_index(text_data)
            total_smog += smog_score
            
            # Lexical diversity
            words = nltk.word_tokenize(text_data.lower())
            lexical_diversity = len(set(words)) / len(words) if words else 0
            total_lexical_diversity += lexical_diversity
            
            # Top-performing words
            fdist = FreqDist(words)
            top_words = fdist.most_common(10)
            
            # Sentiment analysis
            sentiment = analyzer.polarity_scores(text_data)
            neg_sentiment = sentiment['neg']
            neu_sentiment = sentiment['neu']
            pos_sentiment = sentiment['pos']
            compound_sentiment = sentiment['compound']
            total_neg_sentiment += neg_sentiment
            total_neu_sentiment += neu_sentiment
            total_pos_sentiment += pos_sentiment
            total_compound_sentiment += compound_sentiment
            
            # Determine final sentiment
            if compound_sentiment > 0.05:
                final_sentiment = 'Positive'
            elif compound_sentiment < -0.05:
                final_sentiment = 'Negative'
            else:
                final_sentiment = 'Neutral'
            
            # Top-performing CTA words
            cta_word_counts = {word: words.count(word) for word in cta_words}
            
            # "Sales-y" vs "News-y" words
            salesy_count = sum(words.count(word) for word in salesy_words)
            newsy_count = sum(words.count(word) for word in newsy_words)
            total_salesy_count += salesy_count
            total_newsy_count += newsy_count
            
            # Append results to the DataFrame
            new_row = pd.DataFrame({
                'Text': [text_data],
                'Flesch-Kincaid Score': [flesch_kincaid_score],
                'New Dale-Chall': [dale_chall_score],
                'SMOG Index': [smog_score],
                'Lexical Diversity': [lexical_diversity],
                'Top Words': [str(top_words)],
                'Neg Sentiment': [neg_sentiment],
                'Neu Sentiment': [neu_sentiment],
                'Pos Sentiment': [pos_sentiment],
                'Compound Sentiment': [compound_sentiment],
                'Final Sentiment': [final_sentiment],
                'Top CTA Words': [str(cta_word_counts)],
                'Sales-y Words Count': [salesy_count],
                'News-y Words Count': [newsy_count]
            })
            results = pd.concat([results, new_row], ignore_index=True)
        
        # Calculate averages
        num_texts = len(text_data_list)
        averages = pd.DataFrame({
            'Text': ['Averages'],
            'Flesch-Kincaid Score': [total_flesch_kincaid / num_texts],
            'New Dale-Chall': [total_dale_chall / num_texts],
            'SMOG Index': [total_smog / num_texts],
            'Lexical Diversity': [total_lexical_diversity / num_texts],
            'Top Words': ['N/A'],
            'Neg Sentiment': [total_neg_sentiment / num_texts],
            'Neu Sentiment': [total_neu_sentiment / num_texts],
            'Pos Sentiment': [total_pos_sentiment / num_texts],
            'Compound Sentiment': [total_compound_sentiment / num_texts],
            'Final Sentiment': ['N/A'],
            'Top CTA Words': ['N/A'],
            'Sales-y Words Count': [total_salesy_count / num_texts],
            'News-y Words Count': [total_newsy_count / num_texts]
        })
        results = pd.concat([results, averages], ignore_index=True)

        # Display the results
        st.write('Analysis Results:')
        st.dataframe(results)
        
        # Provide explanations
        st.write('### Explanations:')
        st.write('**Flesch-Kincaid Score:** A readability test designed to indicate how difficult a passage in English is to understand. The score is typically between 0 and 12, with higher scores indicating more difficult text. A score of 8-10 is considered fairly difficult, while a score of 12 is very difficult.')
        st.write('**New Dale-Chall:** A readability formula that considers word difficulty and sentence length to assess text difficulty.')
        st.write('**SMOG Index:** A readability formula that estimates the years of education needed to understand a piece of writing. The SMOG index specifically looks for polysyllabic words (words with three or more syllables) and requires a minimum number of sentences to generate a valid score. If the text is too short or lacks sufficient polysyllabic words, the SMOG index may return a score of 0.0. This indicates that the text is either very simple or does not meet the criteria for the SMOG formula to produce a meaningful score.')
        st.write('**Lexical Diversity:** A measure of how many different words are used in the text. It is calculated as the ratio of unique words to the total number of words. A higher ratio indicates a more diverse vocabulary.')
        st.write('**Top Words:** The most frequently occurring words in the text. This helps identify common themes or topics.')
        st.write('**Sentiment Analysis:** An assessment of the emotional tone of the text. The sentiment score ranges from -1 (very negative) to 1 (very positive). A score close to 0 indicates neutral sentiment.')
        st.write('**Top CTA Words:** The most frequently occurring call-to-action words in the text. These words are often used to encourage readers to take specific actions.')
        st.write('**Sales-y Words Count:** The number of words in the text that are typically associated with sales language. A higher count indicates a more promotional tone.')
        st.write('**News-y Words Count:** The number of words in the text that are typically associated with news language. A higher count indicates a more informational tone.')
        
        # Display charts for better visualization
        st.write('### Charts:')
        st.bar_chart(results[['Flesch-Kincaid Score', 'New Dale-Chall', 'SMOG Index', 'Lexical Diversity', 'Sales-y Words Count', 'News-y Words Count']])
        
        # Additional visualizations
        st.write('### Sentiment Distribution:')
        sentiment_df = pd.DataFrame(results[['Neg Sentiment', 'Neu Sentiment', 'Pos Sentiment', 'Compound Sentiment']])
        st.bar_chart(sentiment_df)
        
        st.write('### Top Words Word Cloud:')
        wordcloud_data = {word: total_cta_counts[word] for word in cta_words if total_cta_counts[word] > 0}
        if wordcloud_data:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No CTA words found in the text data to generate a word cloud.")
