import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from annotated_text import annotated_text

from app.run import process_time_range

@st.cache_data(show_spinner=False)
def scrap_articles(api_login, api_password, date_start, date_end):
    ids = []
    leads = []
    texts = []

    date_range = pd.date_range(start=date_start, end=date_end).strftime('%d.%m.%Y').to_list()
    progress_bar = st.progress(0, text="Scrapping data, please wait...")
    for i, date in enumerate(date_range):
        r1 = requests.get(f'https://api.sta.si/news/sl/{date}', auth=(api_login, api_password))
        articles_id = eval(r1.content)

        for id in articles_id:
            r2 = requests.get(f'https://api.sta.si/news/sta/{id}', auth=(api_login, api_password))
            ids.append(id)
            leads.append(r2.json().get('lede', ''))
            texts.append(r2.json().get('text', ''))
        
        progress_percentage = int(i/len(date_range)*100)+1
        progress_bar.progress(progress_percentage, text=f"Scrapping data, please wait... [{i}/{len(date_range)}]")

    progress_bar.progress(100, text="")
    st.success('Articles scrapping completed successfully!', icon="✅")
    df = pd.DataFrame({'id': ids, 'lead': leads, 'text': texts})
    df['content'] = df['lead'] + df['text']

    return df[['id', 'content']]


if __name__ == '__main__': 
    if 'data_scrapped' not in st.session_state:
        st.session_state.data_scrapped = None
      
    st.title('STA - source bias analysis')

    with st.container():
        st.subheader("Articles scrapping")
        api_login = st.text_input('API login', "wroclaw")
        api_password = st.text_input('API password', "k,spa6!z", type='password')
        date_start = st.date_input("Date start:").strftime('%Y.%m.%d')
        date_end = st.date_input("Date end:").strftime('%Y.%m.%d')

        if st.button('Scrap'):
            st.session_state.data_scrapped = scrap_articles(api_login, api_password, date_start, date_end)

    if st.session_state.data_scrapped is not None:
        with st.container():
            st.subheader("Articles preview")
            article_id = st.number_input('Article ID', 0, len(st.session_state.data_scrapped), 0)

            st.text(st.session_state.data_scrapped['content'][article_id])
            annotated_text(
                "This ",
                ("is", "verb"),
                " some ",
                ("annotated", "adj"),
                ("text", "noun"),
                " for those of ",
                ("you", "pronoun"),
                " who ",
                ("like", "verb"),
                " this sort of ",
                ("thing", "noun"),
                "."
            )

            st.text(process_time_range(st.session_state.data_scrapped))




