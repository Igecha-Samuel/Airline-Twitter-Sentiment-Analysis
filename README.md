# Airline Twitter Sentiment Project

![planes](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/6e58fa33-f4a4-4a2b-89e7-dc0a37ebfc35)

# Business Understanding

In the highly competitive airline industry, gaining a competitive advantage and ensuring customer satisfaction are critical for success. To achieve this, understanding customer sentiment and preferences is crucial. By leveraging sentiment analysis on Twitter data, specifically tweets related to airlines, airlines can gain valuable insights into customer feedback and opinions. This understanding allows them to shape strategies, enhance customer satisfaction, and maintain a positive reputation.

Analyzing customer sentiment expressed in tweets provides a comprehensive view of the general perception and public sentiment towards specific airlines. It enables airlines to identify areas of improvement, strengths, and weaknesses in their customer service offerings. By addressing customer concerns and enhancing satisfaction, airlines can foster loyalty and maintain a positive reputation in the highly competitive airline industry.

The importance of sentiment analysis in the airline industry is supported by research, such as the article by the International Air Transport Association IATA. which emphasizes the impact of reputation on an airline's success. By conducting sentiment analysis on Twitter data, airlines can gain valuable insights, identify emerging trends, address negative sentiments, and proactively resolve customer issues, thus mitigating potential reputation risks.

### Objective
The objective of this project is to perform sentiment analysis on tweets related to different airlines and extract valuable insights to enhance customer experiences and maintain a positive reputation.

### 1.1 Specific Objectives
1. Determine the overall sentiment (positive, negative, or neutral) of tweets related to different airlines, providing insights into the general public sentiment towards specific airlines then coming up with solutions for improvement.
2. To use NLP techniques to preprocess and clean text data and prepare it for sentiment analysis.
3. To build multi-classification algorithms that will provide a more comprehensive sentiment analysis of the tweets.
4. To determine the models’ effectiveness in predicting the sentiment of airline passengers based on recall and F1-score metrics.

By achieving these objectives, airlines can gain valuable insights into customer preferences, opinions, and complaints. The analysis will facilitate data-driven decision-making, allowing airlines to improve customer services, address concerns effectively, and build a positive reputation in the competitive airline industry.

The outcome of this project will empower airlines to make informed decisions, enhance customer satisfaction, foster loyalty, and gain a competitive edge by leveraging sentiment analysis to create personalized experiences and strengthen relationships with their customers.

# Data Understanding
Data containing 14640 rows and 20 columns has been obtained from data.world which is more than sufficient and robust enough to be used for our analysis.Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service")

### 1.2 Exploratory Data Analysis

![negatives](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/09bccbef-b915-45bb-8f39-252421deda11)

There were far more negative sentiments in our dataset than positive and negative.

![number of tweets](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/a732e5b1-ce59-47f5-a940-e2ac3f50448d)

Looking at the number of tweets by airline, United had the most meaning they were talked about the most on twitter.

![sentiments by airline](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/f57a6226-0516-4130-bf23-966148d3f134)

United Airline comes first based on the number of the negative, neutral and positive sentiments followed by US Airways then American Airlines. Virgin America comes last

![count reasons](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/07c09dd7-ecc2-47b7-af46-38bb0e42445c)

Upon closer observation of the number of negative reasons of the airlines except for Delta, it looks like customer service issues contributed far more to negative sentiment.

![negative percent](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/assets/54464999/af2bdce2-d016-483b-a8d7-51ce74ebb8c5)

31.7% of tweets indicate poor customer service provided by various airlines. 18.1% indicate a bad experience with late flights and 13.0% of them indicate a dislike the services provided by the airlines without providing any solid reason. Issues concerning damaged luggage are the least.

# Modelling
We will proceed to creating some models .We proceeded with the Neural Network and XGBClassifier models after the vectorization steps, we aimed to improve the accuracy of sentiment classification. The vectorization techniques helped capture the relevant information and patterns within the tweets, enabling the models to learn from these representations and make more informed predictions.

### Convolutional Neural Networks model
The Neural Network model is a deep learning-based approach that utilizes multiple layers of interconnected nodes to learn complex relationships within the data. It consists of an input layer, hidden layers, and an output layer. The network learns from the input features and corresponding labels to optimize its weights and biases, ultimately making predictions on unseen data. The Neural Network model used various activation functions and the Adam optimizer to train the model and minimize the categorical cross-entropy loss. It was trained for a specified number of epochs with a batch size of 32.

# XG Boost Model
XGBClassifier model is based on gradient boosting, an ensemble technique that combines multiple weak predictive models, known as decision trees, to create a strong predictive model. XGBoost stands for "Extreme Gradient Boosting," which optimizes the gradient boosting algorithm to enhance model performance. The XGBClassifier model iteratively builds decision trees, minimizing the loss function through gradient descent. It utilizes a variety of hyperparameters and gradient boosting-specific techniques to improve accuracy and control overfitting.
The XGBClassifier model achieved an accuracy of 0.80 on the given dataset, indicating that it correctly predicted the sentiment of approximately 79.8% of the tweets. The model's performance varied across the different sentiment classes.

For negative sentiment (class 0), the model demonstrated high precision (0.93) and recall (0.83), indicating that it accurately identified the majority of negative tweets. The F1-score for this class was 0.88, reflecting a balanced performance.

For neutral sentiment (class 1), the model's performance was relatively lower. The precision (0.31) and recall (0.56) were both below average, indicating that the model struggled to correctly classify neutral tweets. The F1-score for this class was 0.40, suggesting room for improvement.

Regarding positive sentiment (class 2), the model achieved a precision of 0.64 and a recall of 0.73. The F1-score for this class was 0.69, indicating a relatively balanced performance in identifying positive tweets.

The macro average, which considers the performance across all classes, yielded precision, recall, and F1-score values of 0.63, 0.71, and 0.66, respectively. This suggests a moderate overall performance across the sentiment classes.

The weighted average, which takes into account the class imbalance, resulted in higher precision, recall, and F1-score values of 0.85, 0.80, and 0.82, respectively. This indicates that the model performed well, considering the distribution of sentiment classes in the dataset.

In conclusion, the XGBClassifier model demonstrated the best accuracy on the given dataset, with particularly strong performance for negative sentiment.

# Recommendations
Based on the common words in the tweets with negative and positive sentiments, the airline should focus on improving their customer service, punctuality, and communication with passengers.

For the negative sentiment tweets, the frequent use of words such as "cancelled flight", "delay", and "late flight" suggests that customers are experiencing issues with flight schedules and disruptions, which can be frustrating and stressful. Improving the airline's ability to communicate clearly and promptly with passengers during these disruptions could go a long way towards mitigating negative sentiment. Additionally, addressing the issues with customer service mentioned in the tweets could help to alleviate some of the frustration and dissatisfaction customers are expressing.

For the positive sentiment tweets, the frequent use of words such as "thanks", "great", and "awesome" suggests that customers appreciate good customer service and value when it is provided. Therefore, the airline could continue to focus on improving their customer service and make sure that it is consistent across all interactions with passengers. Additionally, ensuring that flights depart and arrive on time is also likely to be a key factor in generating positive sentiment among customers.

Since a convolutional neural network model has been built to automatically classify airline sentiments, the airline can use this model to gain insights into how customers are feeling about their experiences with the airline. The airline should;

*Focus on improving the customer services by working on customer grievances as soon as needed.
For the late flight, earlier communication of any delays that may be expected may receive a positive feedback.
*By improving the customer care services, this will solve problems with bad flights complaints and this will improve the sentiments towards each airline.

There can be improved luggage process where the airlines can have a team that works to ensure every luggage is encountered for and for every lost or damaged luggage the complaining process should be easy and simple.

*The airline should try and make the checking-in, booking and canceling of flights user friendly for ease access of the services, this can be done by shifting these services online.
*Monitor airline sentiments in real time. By leveraging the machine learning model to monitor sentiment in real-time, the airline can quickly identify and address any issues that arise and deal with them swiftly before causing widespread negative sentiment among customers.

## For More Information See the full analysis in the [Jupyter Notebook](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/blob/main/Airline%20Twitter%20Sentiment.ipynb) or review this [presentation](https://github.com/Igecha-Samuel/Airline-Twitter-Sentiment-Analysis/blob/main/Presentation.pdf)
 
