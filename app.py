import streamlit as st
#from streamlit-option-menu import option_menu
#from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#import preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import string
#from Textblob import TextBlob
from nltk.stem.porter import PorterStemmer
import pickle
from textblob import TextBlob
import sklearn
import io
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')
import numpy as np












selected = st.radio("Select an option:", ["Home"])
f = open("Documents.csv", "rt",encoding="utf8")
#st.button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary", disabled=False, use_container_width=False)
#if st.button("Download File"):
with open("Documents.csv", "r",encoding="utf8") as file:
    file_content = file.read()


def main():
    # Streamlit app title
    #st.title("Download dataset for testing")

    # Add a download button
    st.download_button(
        label="Download Dataset for testing",
        data=file_content.encode('utf-8'),  # Encode the file content to bytes
        file_name="Dataset for testing.csv",  # Name of the downloaded file
        key="download-button"
    )


if __name__ == '__main__':
    main()

if selected =="Home":
    import pandas as pd
    def preprocess(data):
        df=pd.read_csv("Documents.csv")
        
        return df
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.dataframe(df)
    
        
   
    #try:
      
    #if uploaded_file is None:
        #st.warning("Please upload a CSV file.")
    #pass
    #content = uploaded_file.read()
   

    button_clicked = st.button("Show Analysis")

    # Check if the button is clicked
    if button_clicked:
        # Check if 'Review' column exists in the DataFrame
        if 'Review' in df.columns:
            # Combine all reviews into a single string
            reviews_text = ' '.join(df['Review'])

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)
            st.title("Word Cloud of Reviews")
            #st.write("<h3 style='text-align: center; font-size: 16px;'>Word Cloud of Reviews</h3>", unsafe_allow_html=True)

            # Display the word cloud using Streamlit
            st.image(wordcloud.to_image(), caption='Word Cloud of Reviews')
        else:
            st.warning("The 'Review' column does not exist in the DataFrame.")
    #button_clicked = st.button(" Number of times each word used Reviews")
    #if button_clicked:
        # Check if 'Review' column exists in the DataFrame
        if 'Review' in df.columns:
            df=pd.read_csv("Documents.csv")
            stop_words = set(stopwords.words('english'))

            tokenized_reviews = []
            for review in df['Review']:
                word_counts = Counter(tokenized_reviews)
                words = word_tokenize(review)

                #words = words_tokenize(review)
        #print(words)
                words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
                tokenized_reviews.extend(words)
                #st.write("<h3 style='text-align: center; font-size: 16px;'>Most used words Reviews</h3>", unsafe_allow_html=True)

                most_common_words = word_counts.most_common()
            #word_counts = Counter(tokenized_reviews)

            # Convert the results to a DataFrame
            #most_common_words_df = pd.DataFrame(word_counts.most_common(), columns=['Word', 'Count'])
            st.title("Most Used Words in Reviews")
            most_common_words_df = pd.DataFrame(most_common_words, columns=['Words', 'Number of times Words']).head(10)

            # Display the DataFrame in Streamlit
            st.dataframe(most_common_words_df)
        else:
            st.warning("The 'Review' column does not exist in the DataFrame.")
    #button_clicked = st.button(" Scatter Plot ")
    #if button_clicked:
        # Check if 'Review' column exists in the DataFrame
        if 'Review' in df.columns:
            def get_sentiment_polarity(text):
             sentiment = SentimentIntensityAnalyzer()
             sentiment_score = sentiment.polarity_scores(text)['compound']
             return sentiment_score
            st.title("Scatter Plot Showing Polarity Score of Reviews")
            df['Sentiment'] = df['Review'].apply(get_sentiment_polarity)
            fig = px.scatter(df, x=df.index, y='Sentiment',
                        labels={'index': 'Index', 'Sentiment': 'Sentiment'})
            fig.update_yaxes(tickvals=[-1, 0, 1])
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_hline(y=0.5, line_dash="dash", line_color="green")
            fig.add_hline(y=-0.4, line_dash="dash", line_color="purple")

            fig
        if 'RecipeIngredientParts' in df.columns:
          #for i in df["RecipeIngredientParts"]:
            all_text = ' '.join(df["RecipeIngredientParts"].astype(str))

    # Tokenize the words
            tokenized_words = word_tokenize(all_text)

    # Join the tokenized words
            all_words = ' '.join(tokenized_words)

    # Create a word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    # Display the word cloud in Streamlit app
            st.title("Word Cloud for Recipe Ingredients")
    
    # Save the word cloud image to a BytesIO object
            img_buf = io.BytesIO()
            wordcloud.to_image().save(img_buf, format='PNG')
            img_buf.seek(0)

    # Display the word cloud image using st.image()
            st.image(img_buf)
           
    #print(i)
            
          
        if "Rating" in df.columns:
            rating_counts = df['Rating'].value_counts().sort_index()

    # Calculate the percentage of each rating
            rating_percentages = (rating_counts / rating_counts.sum()) * 100
            st.title("Rating Distribution")
    # Create a pie chart using Plotly Express
            fig = px.pie(
            names=rating_percentages.index,
            values=rating_percentages.values,
            title='Rating Distribution'
            #labels={'names': 'Rating', 'values': 'Percentage'}
            )
            fig.update_layout(showlegend=False)
            #fig.update_layout(legend=dict(title=None))
            #fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

            st.plotly_chart(fig)

            #st.fig
    #button_clicked = st.button(" Top ten calories Recipes")
    #if button_clicked:
        
        top_10_recipes = df.sort_values(by='Calories', ascending=False).head(10)
        st.title("Top 10 Recipes with Highest Calories")
        #st.table(top_10_recipes[['AuthorName_y', 'Name', 'Calories']])
        df['Calories'] = np.floor(df['Calories']).astype(int)
        df1=df.sort_values(by='Calories', ascending=False).head(10)

        #columns_to_display = ['Food', 'Calories', 'Protein']
        #sorted_df = df.sort_values(by='Calories', ascending=False)

        columns_to_display = [ 'AuthorName_y', 'Name', 'Calories']
        #st.write("Original DataFrame (Selected Columns):")
        st.write(df1[columns_to_display])

        #st.write(df)
        #st.table(top_10_recipes[['AuthorName_y', 'Name', 'Calories']])
        #print(df["Calories"])
        

        #df['Calories'] = df['Calories'].astype(int)

        fig = px.bar(
        top_10_recipes,
        x='Name',
        y='Calories',
        hover_data=['AuthorName_y','RecipeYield'],
        labels={'Calories': 'Calories', 'Name': 'Recipe Name'}
        #title='Top 10 Recipes with Highest Calories'
    )
        fig.update_xaxes(categoryorder='total descending')


    # Streamlit app
        st.plotly_chart(fig)
        
        selected_recipes = df[(df['Calories'] >= 50) & (df['Calories'] <= 100)]
        top_10_recipes = selected_recipes.sort_values(by='Calories', ascending=True).head(10)
        st.title("Recipes having calories up to 150")
        #fig.update_xaxes(categoryorder='total descending', range=[50, max(top_10_recipes['Calories'])])
        fig = px.bar(top_10_recipes, x='Name', y='Calories', hover_data=['RecipeYield'], labels={'Calories': 'Number of Calories', 'Name': 'Recipe Name'})
        #fig.update_xaxes(categoryorder='total descending', range=[50, max(top_10_recipes['Calories'])])

# Customize the layout
        fig.update_layout(
                  xaxis_title='Recipe Name',
                  yaxis_title='Number of Calories')
        fig.update_xaxes(categoryorder='total descending')
        #fig.update_yaxes(range=[50, 150])
        #fig.update_yaxes(range=[50, 150])
        #fig.update_yaxes(range=[50, max(top_10_recipes['Calories'])])

        st.plotly_chart(fig)

# Show the plot
        #fig.show()


        #top_10_recipes = df.sort_values(by='Calories', ascending=True).head(10)
        #st.title("Top 10 Recipes with Highest Calories")
        #st.table(selected_recipes[['AuthorName_y', 'Name', 'Calories']])
        #fig = px.bar(
        #top_10_recipes,
        #x='Name',
        #y='Calories',
        #hover_data=['AuthorName_y','RecipeYield'],
        #labels={'Calories': 'Calories', 'Name': 'Recipe Name'}
        #title='Top 10 Recipes with Highest Calories'
    #)
        #fig.update_xaxes(categoryorder='total descending')
        #st.plotly_chart(fig)


        


#if selected == "Sentiment Classifier":

import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import string

# Assuming you have your text data in a variable called 'text'
# text = "Your text goes here"
st.title("Sentiment Classifier")
placeholder = st.empty()
input_review = placeholder.text_input("Enter the reviews (required)", key=1)

# Predicting the sentiments of text using the SentimentIntensityAnalyzer
if st.button('Check the Sentiments', key=3):
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Use TextBlob for sentiment analysis
        score_textblob = TextBlob(input_review).sentiment.polarity
        score_textblob = round(abs(score_textblob * 100))

        # Use SentimentIntensityAnalyzer for sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(input_review)
        compound_score =  sentiment_scores['compound']*100
        
        
        print(compound_score)

        if not input_review:
            st.warning("Please fill out the required field")

        elif compound_score >= 0:
            st.subheader("Our Model has Predicted Positive Sentiment")
        elif compound_score <0:
            #compound_score =  np.abs.round(sentiment_scores['compound']*100,2)
            compound_score = np.abs(np.round(sentiment_scores['compound'] * 100, 2))


        
            

        
            st.subheader("Our Model has Predicted Negative Sentiment")

        # Display the sentiment score from SentimentIntensityAnalyzer
        st.subheader(f"Sentiment Score: {compound_score}")

            # Display the score from TextBlob
            #if not input_review:
                #pass
            #elif score_textblob == 0 or 'love' in input_review:
                #st.subheader('TextBlob Score: 100%')
            #else:
                #st.subheader(f'TextBlob Score: {score_textblob}%')

            
