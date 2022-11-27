# Architecture and Deployment of a Scalable MachineLearning Pipeline in a Distributed Cloud Environment

## Abstract
The architecture and deployment of complex machine learning (ML) pipelines for big data use
cases pose many technical design challenges. NET CHECK, the company this thesis was conuducted in cooperation with, faces several issues with their existing data architecture and 
therefore seeks to re-engineer their data and Machine Learning pipelines. This thesis investiagates tools, platforms, and frameworks best suited to build scalable, reliable, and maintainaable batch prediction pipelines in distributed cloud environments. Specifically, Apache Spark is 
used for data ingestion, data transformation, and training of the ML models, while Apache 
Airflow is utilized to orchestrate the individual components of the pipeline. The resulting application is then deployed on a Kubernetes cluster in a Microsoft Azure cloud instance. First,
the identified tools and methods are described in detail before an application architecture 
build according to the research findings will be conceptualized. Subsequently, the implementation of a proof-of-concept application follows the steps dictated by the CRISP-ML(Q) process 
model. In conclusion, the deployed pipelines are evaluated according to their scalability, reliability, and maintainability, confirming that the re-development outperforms the monolithic 
legacy application in all categories.

![image](https://user-images.githubusercontent.com/58426096/204164200-06848559-e709-437c-a729-9547db72036e.png)

