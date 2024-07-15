<a name="readme-top"></a>

<div align='center'>
    <h1><b>StocKnock</b></h1>
    <img src='pic/companyLogo.png'/>
    <br><br>
    <p>
    This project focuses on developing an application to predict the stock performance of several electric vehicle manufacturers based on sentiment analysis from Tesla news articles on Twitter.
    </p>
    <br>

![Python](https://badgen.net/badge/Python/3.9.18/blue?)
![Streamlit](https://badgen.net/badge/Streamlit/1.10.0/orange?)
![Pandas](https://badgen.net/badge/Pandas/1.4.3/blue?)
![Seaborn](https://badgen.net/badge/Seaborn/0.11.2/green?)
![Matplotlib](https://badgen.net/badge/Matplotlib/3.5.2/blue?)
![Scikit-learn](https://badgen.net/badge/scikit-learn/1.4.2/yellow?)
![Plotly](https://badgen.net/badge/Plotly/5.22.0/cyan?)
![TensorFlow](https://badgen.net/badge/TensorFlow/2.15.0/orange?)
![WordCloud](https://badgen.net/badge/WordCloud/1.8.1/purple?)
![NLTK](https://badgen.net/badge/NLTK/3.7/red?)
![Docker](https://badgen.net/badge/Docker/20.10/cyan?)

</div>

---

## 🧑‍💻 **Team Members**

- **Reski Hidayat**
  - Role: Data Engineer & Data Scientist 
  - [Github](https://github.com/Reaumurr) | [LinkedIn](https://www.linkedin.com/in/reskihidayat/)

- **Fadhil Athallah**
  - Role: Data Scientist  
  - [Github](https://github.com/FadhilAthallah) | [LinkedIn](https://www.linkedin.com/in/fadhil-athallah-297876237/)
  
- **M. Gifhari Heryndra**
  - Role: Data Analyst  
  - [Github](https://github.com/heryndra) | [LinkedIn](https://www.linkedin.com/in/m-gifhari-heryndra-45886a87/)

<br />

## 💾 **Dataset**

The dataset is obtained from a credible source and comprises relevant details regarding Tesla News from Twitter. For further information or to access the sentiment dataset, please refer to the provided source [here](https://www.kaggle.com/datasets/drlove2002/tesla-news-from-tweeter). For Electric Vehicle Stocks dataset we refer to Yahoo! Finance.

<br />

## ⚠️ **Problem Statement**

In recent years, the electric vehicle (EV) industry has experienced significant growth and has become a focal point for investors seeking to capitalize on the transition to sustainable energy. However, the volatile nature of the stock market, especially within emerging industries like EVs, poses a considerable challenge for investors. Accurately predicting stock movements in the EV sector requires sophisticated analytical tools and methods.

StocKnock aims to address this challenge by developing a machine learning application that provides accurate and actionable predictions for EV stocks, with a particular focus on incorporating Twitter sentiment analysis related to Tesla news. As Tesla is a leading player in the EV market, public sentiment surrounding its developments can greatly influence investor perception and stock performance. By analyzing Twitter data, we can gauge market sentiment and its potential impact on Tesla's stock movements.

Our goal is to empower investors with reliable insights, thereby reducing uncertainty and enabling informed investment decisions in the electric vehicle ecosystem. By leveraging advanced machine learning techniques, StocKnock will analyze historical stock data, market trends, and various economic indicators, alongside real-time sentiment analysis from social media platforms, to forecast the performance of EV stocks.

Through this project, we seek to revolutionize the way investors approach the EV market, offering a cutting-edge solution that marries technology with investment strategy. Join us in shaping the future of investment in the electric vehicle industry.

<br />

## 📌 **Objective**

The primary goal of StocKnock is to enhance investment decision-making in the electric vehicle (EV) sector by providing personalized stock predictions based on sentiment analysis of Tesla news articles on Twitter. Specifically, StocKnock aims to:

* Analyze sentiment from Twitter discussions regarding Tesla news to classify them as positive, neutral or negative, with accuracy being the key performance metric.
* Predict stock movements for various EV manufacturers based on identified sentiment trends, offering users tailored investment insights.
* Recommend potential investment opportunities in EV stocks that exhibit strong positive sentiment, helping investors capitalize on favorable market conditions.
* Provide alerts for underperforming stocks with negative sentiment, guiding users toward better investment alternatives in the EV sector.

<br />

---

## 🗒️ **Setup and Installation**

To get started with FlightBuddy, ensure you have the following prerequisites:

- **Dataset**: Yahoo! Finance and Tesla News Sentiment Accessible [here](https://www.kaggle.com/datasets/drlove2002/tesla-news-from-tweeter).
- **Python**: Version 3.9.18 or later.

### **Environment Configuration**  
Ensure you have all necessary Python packages by installing them from the provided `requirements.txt`.

### **Project Setup**  
Follow these steps to set up the project:

1. **Clone the Repository**
   Clone this repository to your local machine. Choose the method that best suits your setup:
   - **HTTPS**:
     ```
     git clone https://github.com/FTDS-assignment-bay/p2-final-project-sarjana-kakek/
     ```
   - **SSH**:
     ```
     git clone git@github.com:FTDS-assignment-bay/p2-final-project-sarjana-kakek.git
     ```

2. **Compose Docker Containers (Optional)**  
   If you prefer using Docker, build and run the Docker container as follows:
```
docker build -t stocknock-app .
docker run -it stocknock-app
```


3. **Environment Setup**  
- Navigate to the cloned directory:
  ```
  cd p2-final-project-sarjana-kakek
  ```
- Set up a virtual environment (optional but recommended):
  ```
  python -m venv venv
  source venv/bin/activate  # On MacOS/Linux
  .\venv\Scripts\activate   # On Windows
  ```
- Install the required dependencies:
  ```
  pip install -r requirements.txt
  ```

4. **Run the Application**  
Execute the main application script:
```
python app.py
```

5. **Access and Use**  
After starting the application, you can access and interact with it as specified in your project documentation.

### **Additional Resources**  
For further exploration or modifications, access the full project documentation and source code on the [GitHub repository](https://github.com/FTDS-assignment-bay/p2-final-project-sarjana-kakek/).

By following these setup instructions, you'll be able to replicate the FlightBuddy project and explore its functionalities related to analyzing airline review sentiments.

---

## 💻 **Tools and Libraries**

![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-%238DD6F9.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23D00000.svg?style=for-the-badge&logo=matplotlib&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-%232376C6.svg?style=for-the-badge&logo=nltk&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![WordCloud](https://img.shields.io/badge/WordCloud-%23FF8800.svg?style=for-the-badge&logo=wordcloud&logoColor=white)
![TextBlob](https://img.shields.io/badge/TextBlob-%23157AF6.svg?style=for-the-badge&logo=textblob&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)

<br />

## 🔄 **Workflow**
- Data Analyst (DA): Responsible for creating visualizations to provide insights from the data.
- Data Scientist (DS): Develops the NLP, Time Series, Regression model, and handles deployment using Streamlit.
- Data Engineer (DE): Manages databases, manipulates data in PostgreSQL, and schedules tasks using Elasticsearch.

<br />

## 📂 **File Descriptions**
- eda_all.ipynb: Contains the Exploratory Data Analysis.
- stock_predict.ipynb: Notebook for developing the Regression, Time Series, and NLP model, including training, preprocessing and inference.
- twitter_sentiment.ipynb: Notebook for developing NLP model, including training, preprocessing and inference.

<br />

## 🚀 **Deployment**
The application is deployed on Hugging Face Spaces. Access it using the following link:
[StocKnock on Hugging Face](https://huggingface.co/spaces/Reaumur/StocKnock)

<p align="right">(<a href="#readme-top">back to top</a>)</p>





