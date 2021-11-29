
Classification Project 

About the Project

Project Goals

    The goal of this project is to create a classification module. This module can generate a list of customers who have high probability to churn by identifing main drivers of churn. For this project we have access to codeup database (telco_churn) from where we can pull the data. 


Project Description

    This project provides us the opportunity to implement our knowledge in statistics, SQL, python programming, data science libraries and classification module. We will creating and evaluating various predictive models that are part of data science pipeline. 
    In order to complete this project we have to go through various processes such as acquiring Telco data from SQL database; observing data; cleaning data; preparing data; create functions; splitting data into train ,test and validate;  exploratory data analysis; modeling; model evaluation; model testing and presenting our findings and recommendations. 

    Due to high competition in telecommunication business there is a higher churn rate, our primary task for this project is to identify drivers of higher churn and provide possible recommendation to reduce churn rate. 



Initial Questions

    Where do we acquire the data?
    What are the main driver of Telco churn?
    Are there any duplicates in Data?
    Are there any misssing values in data ?
    Are customers with DSL more or less likely to churn?
    What month are customers most likely to churn and does that depend on their contract type?
    Is there a service that is associated with more churn than expected?
    Do customers who churn have a higher average monthly spend than those who don't?


Data Dictionary:

    customer_id: unique identifier for each customer
    senior_citizen: (int) 1 indicates the customer is a senior citizen, 0 indicates they are not
    partner: (int) 1 indicates they have partners, 0 indicates they don’t have partners
    dependents: (int) 1 indicates they have dependents, 0 indicates they don’t have dependents
    tenure: (int) length customer has been with the company in months
    phone_service: (int) 1 indicates they have phone service, 0 indicates they don’t have phone service
    multiple_lines: (int) 1 indicates they have multiple lines, 0 indicates that they don’t
    online_security: (int) 1 indicates customers opted in for online security service, 0 indicates they havent
    device_protection: (int) 1 indicates customers have device protection, 0 indicates they dont
    tech_support: (int) 1 indicates customers have tech support, 0 indicates they dont
    streaming_movies: (int) 1 indicates customers have streaming movie service, 0 indicates they dont
    paperless_billing: (int) 1 indicates customers have enrolled in paperless billing, 0 indicates they havent
    monthly_charges: (int) monthly charge of a customer
    total_charges: (float) total charges customers have paid
    churn: (int) 1 to represent customers that have churned, 0 represent they havent
    partner_dependents: (bool) to represent if customers have both partner and dependents
    phone_and_multiple_lines: (bool) to represent if customers have both phone and multiple lines
    streaming_tv_movie: (bool) to represent if customers have both streaming tv and movie
    online_security_and_backup: (bool) to represent if customers have both online security and back up
    payment_auto: (int) 1 indicates customers have automatic payment, 0 indicates they have not
    month-to-month: (int) 1 indicates customers are with month to month, 0 indicates they have other form of contract
    one year: (int) 1 indicates customers are with one year, 0 indicates they have other form of contract
    two year: (int) 1 indicates customers are with two year, 0 indicates they have other form of contract
    DSL: (int) 1 indicates customers have DSL, 0 indicates they have other form of internet
    fiber optic: (int) 1 indicates customers have fiber optic, 0 indicates they have other form of internet


Project Planning

Data Acqusition:

     Acquire data from CodeUp database. You are required to have credentials to access the data base.
     Create a cache in your local repository in order to prevent from having to access the database every time you re-work on the project
     Create acquire.py for reproducibility


Data Preparation:

     Summarize data
     Check for duplicates, removed duplicates if present
     Check for null values, missing values, and get count of them
     Check the data types
     Perform feature engineering
     Perform data types conversion/Use Dummy variables
     Drop redundant/useless columns
     Split data into Train, Test, and Validate; ~50%-30%-20% split train-validate-test; 
     Create prepare.py



Data Exploration:

     Explore various features and relationships
     Vizualize data
     Impute missing values
     Test different hypothesis

Model Building

     Try various algorithms(Logit, Random Forest, KNN, Decision Tess) on Train data
     Perform model evaluation
     Pick top 3 perfomring models
     Validate top 3 models on validate data and evaluate their performance
     Pick best performing model on validate data
    Use the top model with test data set and evaluate your model


Conclusion

Summarize the process
Summarize findings
Make recommendations





code commenting (Report.ipynb)

Your code contains code comments that are helpful to the reader in understanding what each blocks/lines of code are doing.
markdown (Report.ipynb)

Notebook contains adequate markdown that documents your thought process, decision making, and navigation through the pipeline. This should be present throughout the notebook consistently, wtih not just headers, but plenty of content that guides the reader and leaves no questions or doubt as to why you did something, e.g.
Written Conclusion Summary (Report.ipynb)

Your conclusion summary should addresses the questions you raised in the opening of the project, which we would want to see at the end of every final notebook. Ideally, when the deliverable is a report, the summary should tie together your analysis, the drivers of the outcome, and how you would expect your ML model to perform in the future on unseen data, in layman's terms.
conclusion recommendations (Report.ipynb)

Your notebook should ends with a conclusion that contains actionable recommendations based on your insights and analysis to the business stakeholder(s), your simulated audience, or someone who would find this information valuable (if there is no stakeholder). Your recommendations should not be not about what to do differently with the data, but instead should be based on the business or domain you are studying.
conclusion next steps (Report.ipynb)

Your conclusion should include next steps from a data science perspective that will assist in improving your research. Ideally, if you talk about trying more algorithms to improve performance, think about why you need to improve performance. And if the business calls for it, remember the best way to improve performance is to have better predictors/features. If you talk about gathering more data, being specific about what data you think will help you understand the problem better and why is the way to go!
no errors (Report.ipynb)

Your final notebook should run without error. One error in a notebook can lead to the rest of it erroring out. If you have a reader who doesn't know python, they will then not be able to consume your report.
Live Presentation

intro (live)

Speaker kicks of the presentation by introducing themselves and their project through a one-liner of what it's about.
audience & setting (live)

Always be aware of the audience and setting for your presentation. What is the appropriate level of technicality? What is the appropriate depth given audience, setting and medium in which its delivered. The way you communicate should be appropriate for the audience: volume, speed of talk, flow, professionalism. (Codeup Data Science Instructor Team, virtually delivered via jupyter notebook).
content (live)

Notebook talked through step-by-step, in an understandable and meaningful way. Extraneous content in the notebook is not present.
Verbal Conclusion (findings, next steps, recommendations) (live)



Deliver predictions (.csv)

A csv with predictions made from the top model developed should be submitted, as per instructions in the project spec.

