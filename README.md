# Topic Modeling Analysis of Yelp Reviews

<div align="center">

```
          ⚡ 
         ⚡⚡
        ⚡⚡⚡
       ⚡⚡⚡⚡
      ⚡⚡⚡⚡⚡
     ⚡⚡⚡⚡⚡⚡
    ⚡⚡⚡⚡⚡⚡⚡
   ▀▀▀▀▀▀▀▀▀▀▀
```

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-success)

*Unveiling Hidden Patterns in Customer Reviews Using Advanced NLP*

</div>

## 📊 Project Overview

Analysis of 4,885 Yelp reviews using topic modeling to uncover hidden patterns and themes in customer feedback. Through LDA (Latent Dirichlet Allocation), NMF, and LSA approaches, we identify key discussion topics and their relationships.

## 🔍 Key Findings

### Topic Distribution Network

```mermaid
graph LR
    T3((Topic 3)) --- T8((Topic 8))
    T3 --- T4((Topic 4))
    T8 --- T4
    T9((Topic 9)) --- T3
    T9 --- T8
    T9 --- T4
    
    style T3 fill:#0000FF,color:#fff
    style T8 fill:#0000FF,color:#fff
    style T4 fill:#0000FF,color:#fff
    style T9 fill:#E6E6FA
```

### Word Frequency Analysis

```
Most Frequent Terms in Reviews
────────────────────────────────────────
good    ████████████████████████  2546
great   ██████████████████████   2083
service ████████████████████     1939
time    ███████████████████      1697
back    ████████████████         1349
order   ███████████████          1287
chicken ██████████████           1078
pizza   ████████████              974
also    ███████████               875
came    ███████████               851
```

### Topic Composition

```mermaid
pie title Topic Distribution
    "Restaurant Experience" : 15.2
    "Food Quality" : 14.1
    "Golf/Sports" : 12.3
    "Filipino Cuisine" : 10.7
    "Latin/German" : 9.8
    "Others" : 37.9
```

## 📈 Model Performance

### Coherence Scores
```
Model Evaluation Metrics
─────────────────────────────────────
LDA  │████████████▌│ Score: 0.513
NMF  │████████████▌│ Score: 0.512
LSA  │██████████▌  │ Score: 0.445
─────────────────────────────────────
```

### Topic Stability
```python
Stability Scores:
┌─────────┬────────────┬───────────┐
│ Model   │ Stability  │ Variance  │
├─────────┼────────────┼───────────┤
│ LDA     │ 0.627      │ ±0.021    │
│ NMF     │ 0.712      │ ±0.072    │
│ LSA     │ 0.684      │ ±0.070    │
└─────────┴────────────┴───────────┘
```

## 🛠️ Implementation

### Processing Pipeline
```mermaid
graph TD
    A[Raw Reviews: 4,885] -->|Language Detection| B[English Reviews]
    B -->|Text Cleaning| C[Preprocessed Text]
    C -->|Vectorization| D[Document-Term Matrix]
    D -->|Topic Modeling| E[10 Topics]
    E -->|Analysis| F[Insights]
    
    style A fill:#f9f,stroke:#333
    style F fill:#bf9,stroke:#333
```

### Vocabulary Statistics
```
Unique Terms Analysis
──────────────────────────────
Unigrams  │██████████│ 12,910
Bigrams   │██████████│ 141,718
Trigrams  │██████████│ 189,469
──────────────────────────────
```

## 🔬 Key Insights

1. **Topic Distribution**
   ```
   Most Prevalent Topics
   ────────────────────────────────────
   Restaurant Experience ████████ 15.2%
   Food Quality        ███████  14.1%
   Golf/Sports        ██████   12.3%
   Filipino Cuisine   ██████   10.7%
   Latin/German       █████     9.8%
   ────────────────────────────────────
   ```

2. **Common Bigrams**
   ```mermaid
   graph LR
       A[first time] --- B[come back]
       B --- C[customer service]
       C --- D[great service]
       D --- E[highly recommend]
   ```

## 👤 Author

<div align="center">

### Mohamed Amine EL MOUSSAOUI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/medaminelmoussaoui/)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/MOSSAWIII)

</div>

## 📚 Technologies Used

```mermaid
graph LR
    A[Python 3.8+] --> B[NLTK]
    A --> C[spaCy]
    A --> D[scikit-learn]
    A --> E[Gensim]
    
    style A fill:#326DE6
    style B,C,D,E fill:#2196F3
```

## 📈 Future Improvements

- [ ] Implementation of dynamic topic modeling
- [ ] Integration of sentiment analysis
- [ ] Development of interactive visualizations
- [ ] Enhanced topic interpretation guidelines

---

<div align="center">

Made with 💻 by Mohamed Amine EL MOUSSAOUI

*If you found this project helpful, please consider giving it a ⭐*

</div>
