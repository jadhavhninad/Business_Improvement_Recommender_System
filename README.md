## [](#header-1)Business Improvement Recommender System

### [](#header-3) Goals:
*   Success of any business greatly depends on the customer experience related to numerous services that a business offers. Many business use the traditional method of getting customer feedback by sending survey requests via email or by maintaining a feedback book at the business outlet. Such methods fall short of documenting actual user experience since most of time people do not bother with filling long surveys. 
*   On the other hand the very same users have high social media activity and generally leave comments and ratings on websites like Yelp.Such reviews have a strong impact on the success of businesses as an extra half-star rating causes restaurants to sell out 19 percentage points more frequently (increase from 30% to 49%). But it can be difficult to pinpoint what a business is doing wrong unless someone manually reads the reviews. 
*   In this project we provide a faster approach that combines different machine learning techniques to extract most frequently discussed negative topics/keywords that will provide business with insights regarding what they should be doing right based on what they are currently doing wrong. We use sentiment analysis to extract features from negative reviews to get higher business specific accuracy.

### [](#header-3) Implementation:
*   Using Neural Networks (LSTM, CNN) and SVM, a _Sentiment Analysis_ module was developed to separate positve and negative yelp reviews for a specific business
*   Topic modelling done on the negative reviews to extract latent topics using _Latent Dirichlet Allocation_
*   Extracted the most common keywords among the topics and re-iterated the topic modelling for those keyword specific texts in the negative reviews to filter out false positives.
*   Top 5 Final recommendations were given to business ordered by the most negative aspect that needs improvement 
*   The program was scaled up to handle multiple outlets for a specific business eg: _Starbucks_

### [](#header-3) Architecture:

!["Implementation Architecture"](https://github.com/jadhavhninad/Business_Improvement_Recommender_System/blob/master/assets/images/fd.png)

### [](#header-3) Level 1 Topic Modelling (with False positives):

!["Topic Modelling"](https://github.com/jadhavhninad/Business_Improvement_Recommender_System/blob/master/assets/images/itm.png)

### [](#header-3) Sample Output:

!["Ouput"](https://github.com/jadhavhninad/Business_Improvement_Recommender_System/blob/master/assets/images/Sample_output.png)

<br><br>
