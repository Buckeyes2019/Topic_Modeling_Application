#%%

###### Topic_Modeling_Streamlit_App ######

import streamlit as st
import pandas as pd
from texthero.representation import most_similar
import textacy
import textacy.tm
import spacy
from functools import partial
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# %%
st.set_page_config(page_title="Topic Modeling Analysis Tool")

st.title("Topic Modeling Analysis Tool")

st.write("_This application is for identifying topics in large bodies of text._")

st.write("To begin, please upload a CSV file containing the text you would like to analyze. Below is an example of a properly formatted CSV file. The first column should contain the text you would like analyzed. Additional columns, such as 'Source' and 'Date' in the example below, are optional.")

demo = {'Text': ['This is the first sentence.', 'And this is the second', 'And this is the third.'], 'Source': ['Internet', 'Newspaper', 'TV'], 'Date': ['2/13/2021','4/27/2021','8/1/2021']}
demo1 = pd.DataFrame(demo)
st.dataframe(demo1)

dfs = st.file_uploader("Please drag and drop your file into the space below (.csv files only)", type=["csv"])

st.sidebar.write("Below are several options for displaying your results. Feel free to experiment with different settings.") 

most_common_words = st.sidebar.slider('Number of Most Frequent Words to Display', min_value=1, max_value=20, value=10, step=1)
#check_1 = st.sidebar.checkbox('Show Chart of Most Frequent Words')
num_topics = st.sidebar.slider('Number of Topics to Find ', min_value=1, max_value=12, value=6, step=1)
words_in_topic = st.sidebar.slider('Number of Words Per Topic to Display', min_value=1, max_value=10, value=5, step=1)
n_grams = st.sidebar.slider('N-Gram Size (single words, two word phrases, or three word phrases)', min_value=1, max_value=3, value=2, step=1)

#st.sidebar.write('Select Topic Modeling Algorithm')
model_type = st.sidebar.radio('Select Topic Modeling Algorithm', ['Latent Dirichlet Allocation','Non-Negative Matrix Factorization', 'Latent Semantic Analysis'])
if model_type == 'Non-Negative Matrix Factorization':
    model_type = 'nmf'
elif model_type == 'Latent Dirichlet Allocation':
    model_type = 'lda'
elif model_type == 'Latent Semantic Analysis':
    model_type = 'lsa'
#idf_type1 = st.sidebar.radio('Inverse Document Frequency Type', ['standard', 'smooth', 'bm25'])

graph_topic_to_highlight = [0,2,4]

check_2 = st.sidebar.checkbox('Show Chart of Topics and Words')
graph_num_words = st.sidebar.slider('Words to Display on Topic Chart', min_value=1, max_value=20, value=6, step=1)

submit = st.sidebar.button('Analyze Text')

if submit:
    df = pd.read_csv(dfs)
    first_column = df.iloc[1:, 0] 
    texts = first_column.to_list()
    test = [str(x) for x in texts]
  
    corpus = textacy.Corpus("en_core_web_sm", data=test)
    word_doc_counts = corpus.word_doc_counts(by="lemma_", weighting="freq")
    top_ten_words = sorted(word_doc_counts.items(), key=lambda x: x[1], reverse=True)[:most_common_words]
    docs_terms = (textacy.extract.terms(doc, ngs=partial(textacy.extract.ngrams, n=n_grams, include_pos={"PROPN", "NOUN", "ADJ", "VERB"})) for doc in corpus)#, ents=partial(textacy.extract.entities, include_types={"PERSON", "ORG", "GPE", "LOC"})) 
    tokenized_docs = (textacy.extract.terms_to_strings(doc_terms, by="lemma") for doc_terms in docs_terms)
    doc_term_matrix, vocab = textacy.representations.build_doc_term_matrix(tokenized_docs, tf_type='linear', idf_type='smooth')
    model = textacy.tm.TopicModel(model_type, n_topics=num_topics)
    model.fit(doc_term_matrix)
    doc_topic_matrix = model.transform(doc_term_matrix)
    id_to_term = {id_: term for term, id_ in vocab.items()}

    st.write('### The most frequent words in the text are:')
    variable1 = most_common_words//2
    variable2 = most_common_words-variable1
    col1, col2 = st.columns(2)
    counter = 0
    for i in range(variable2):
        counter += 1
        col1.write(f"{i+1}. {top_ten_words[i][0]} - {top_ten_words[i][1]*100:.2f}%")
    for i in range(variable1):
        col2.write(f"{counter+1}. {top_ten_words[counter][0]} - {top_ten_words[counter][1]*100:.2f}%")
        counter += 1
    test_list = []
    st.write('### The topics identified are:')
    for topic_idx, terms in model.top_topic_terms(id_to_term, top_n=words_in_topic):
        st.write(f"**Topic {topic_idx+1}**: {'; '.join(terms)}")
    if check_2:
        st.write('### Chart of Topics and Words')
        plot1 = model.termite_plot(doc_term_matrix, id_to_term, n_terms=graph_num_words, highlight_topics=graph_topic_to_highlight,save = "termite_plot.png")
        st.image('./termite_plot.png')

# %%
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            footer:after {
	        content:'Created by AJ'; 
	        visibility: visible;
	        display: block;
	        position: relative;
	        padding: 5px;
	        top: 2px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 