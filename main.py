import streamlit as st

st.set_page_config(page_title="The Intelligent Project", page_icon="🤖", layout="centered")

st.title("The Intelligent Project")
st.subheader("- Machine Learning")
st.subheader("HDPredictor")

st.image("https://hdmall.co.th/blog/wp-content/uploads/2024/09/cardiovascular-treatment-comparison-01-scaled.jpg.webp", width=650)

st.markdown(
    """
    <a href="/Machine_Learning" style="text-decoration: none;">
        <div class="mainnaural">
            <h3 style="text-shadow: 2px 2px 5px black; text-align: center; color:white; background-color: red; padding: 10px; border-radius: 5px; font-size: 20px;">Start HDPredictor!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)

st.subheader("- Neural Network")
st.subheader("Cat vs Dog Classification")

st.image("https://i.ytimg.com/vi/PV63uCaW8dc/maxresdefault.jpg", width=700)

st.markdown(
    """
    <a href="/Neural_Network" style="text-decoration: none;">
        <div class="mainnaural">
            <h3 style="text-shadow: 2px 2px 5px black; text-align: center; color:white; background-color: red; padding: 10px; border-radius: 5px; font-size: 20px;">Start Cat vs Dog Classification!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)
