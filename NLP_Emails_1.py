
# coding: utf-8

# # Women Who Code
# # NLP exercise - Hillary Clinton's Emails Subject Analysis
# 
# ## Exploratory Analysis: Getting and Cleaning Data
# 

# In[1]:

# Loading libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Emails.csv
# 
# 1. Id - unique identifier for internal reference
# 2. DocNumber - FOIA document number
# 3. MetadataSubject - Email SUBJECT field (from the FOIA metadata)
# 4. MetadataTo - Email TO field (from the FOIA metadata)
# 5. MetadataFrom - Email FROM field (from the FOIA metadata)
# 6. SenderPersonId - PersonId of the email sender (linking to Persons table)
# 7. MetadataDateSent - Date the email was sent (from the FOIA metadata)
# 8. MetadataDateReleased - Date the email was released (from the FOIA metadata)
# 9. MetadataPdfLink - Link to the original PDF document (from the FOIA metadata)
# 10. MetadataCaseNumber - Case number (from the FOIA metadata)
# 11. MetadataDocumentClass - Document class (from the FOIA metadata)
# 12. ExtractedSubject - Email SUBJECT field (extracted from the PDF)
# 13. ExtractedTo - Email TO field (extracted from the PDF)
# 14. ExtractedFrom - Email FROM field (extracted from the PDF)
# 15. ExtractedCc - Email CC field (extracted from the PDF)
# 16. ExtractedDateSent - Date the email was sent (extracted from the PDF)
# 17. ExtractedCaseNumber - Case number (extracted from the PDF)
# 18. ExtractedDocNumber - Doc number (extracted from the PDF)
# 19. ExtractedDateReleased - Date the email was released (extracted from the PDF)
# 20. ExtractedReleaseInPartOrFull - Whether the email was partially censored (extracted from the PDF)
# 21. ExtractedBodyText - Attempt to only pull out the text in the body that the email sender wrote (extracted from the PDF)
# 22. RawText - Raw email text (extracted from the PDF)

# In[2]:

# Loading data into dataframe

emails = pd.read_csv("~/Documents/WWC/NLP_PYTHON/Emails.csv")



# In[3]:

emails.head()


# In[4]:

cols = ['Id', 'DocSubject', 'To', 'From', 'PersonId','DateSent', 'DateReleased', 'pdfLink',
        'CaseNumber', 'DocClass' 'pdfSubject', 'pdfTo', 'pdfFrom', 'pdfCc', 'pdfDateSent',
        'pdfCaseNumber', 'pdfDocNumber', 'pdfDateReleased', 'RinPartorFull', 'pdfBodyTest',
        'pdfRawEmail', 'x']

no_headers = pd.read_csv('~/Documents/WWC/NLP_PYTHON/Emails.csv', sep=',', header=0,
                         names=cols)


# In[5]:

no_headers.head() 


# In[6]:

no_headers.ndim # Display DataFrame attributes (Number of dimensions)


# In[7]:

no_headers.shape # Number of elements in the dataframe


# In[8]:

no_headers.dtypes # Types of elements


# ### Slicing Dataframe to extract Subject ###
# 
# *Object > Type Selection >	  Return Value Type*  
# 
# Series	 >  series[label] > scalar value  
# 
# DataFrame > frame[colname] >  Series corresponding to colname  
# 
# Panel >  panel[itemname] > DataFrame corresponding to the itemname

# In[9]:

emailsSubjects = no_headers['DocSubject']


# In[10]:

emailsSubjects[0:5]  


# In[11]:

emailsSubjects.dtypes


# In[12]:

emailsSubjects.size


# In[13]:

emailsSubjects.ndim 


# ## Pre-Processing Data ##
# 
# The NLTK module is a Python kit to perform Natural Language Processing (NLP). With NLTK you would be able to split sentences from paragraphs, split up words, recognize the part of speech of those words, and highlight the main subjects. In this series, we're going to focus on subject/topic mining and sentiment analysis.
# 
# ### Vocabulary ###
# 
# Corpus - Body of text, singular. Corpora is the plural of this. Example: A collection of medical journals.
# 
# Lexicon - Words and their meanings. Example: English dictionary. Consider, however, that various fields will have different lexicons. For example: To a financial investor, the first meaning for the word "Bull" is someone who is confident about the market, as compared to the common English lexicon, where the first meaning for the word "Bull" is an animal. As such, there is a special lexicon for financial investors, doctors, children, mechanics, and so on.
# 
# Token - Each "entity" that is a part of whatever was split up based on rules. For examples, each word is a token when a sentence is "tokenized" into words. Each sentence can also be a token, if you tokenized the sentences out of a paragraph.
# 
# From:https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
# 
# ### Cleaning ###
# 
# 1. Eliminating punctuation
# 2. Eliminating stopwords
# 2. Normalizing data: converting to lower case
# 3. Tokenizing words
# 

# In[14]:

import nltk 


# In[15]:

type(emailsSubjects)


# In[16]:

Subject = emailsSubjects[2]


# In[17]:

nltk.sent_tokenize(Subject)


# In[18]:

nltk.word_tokenize(Subject)


# In[19]:

from nltk.corpus import stopwords


# In[20]:

set(stopwords.words('english'))


# In[21]:

stopWords = set(stopwords.words('english'))


# In[22]:

tokens = nltk.word_tokenize(Subject) # Getting all the words within the subject


# In[23]:

type(tokens)


# In[24]:

tokens


# In[25]:

mylist = emailsSubjects[0:5]
mylist


# In[26]:

tokenStrg = '\n'.join(map(str, mylist))
tokenStrg


# In[27]:

type(tokenStrg)


# In[28]:

from nltk.tokenize import RegexpTokenizer # Eliminating punctuation

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(tokenStrg)


# In[29]:

tokens


# In[30]:

cleanup = [token.lower() for token in tokens if token.lower() not in stopWords and  len(token)>2]


# In[31]:

cleanup # Display normalized tokens in slice 0:5


# ### Exploring Data using a WordCloud

# In[32]:

type(cleanup)


# In[33]:

tokenStrgCln = ' '.join(map(str, cleanup))
tokenStrgCln


# In[34]:

tokensCln = tokenizer.tokenize(tokenStrgCln)


# In[35]:

set(tokensCln)


# In[36]:

import wordcloud as wc


# In[37]:

from wordcloud import WordCloud, STOPWORDS


# In[38]:

import matplotlib.pyplot as plt


# In[39]:

#Convert all the required text into a single string here 
#and store them in word_string (tokenStrgCln)

#you can specify stopwords, background color and other options

wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',width=1200,height=1000).generate(tokenStrgCln)


# In[40]:

plt.imshow(wordcloud)


# In[41]:

plt.axis('off')


# In[ ]:

plt.show()


# ### Frequency Distribution
# 
# How can we automatically identify the words of a text that are the most informative about the topics on Hillary Clinton's emails?
# 
# 

# In[42]:

from nltk.probability import *


# In[43]:

tokens = nltk.word_tokenize(tokenStrgCln)


# In[44]:

fdist = nltk.FreqDist(tokens)


# In[45]:

fdist


# In[46]:

vocabulary = fdist.keys()


# In[47]:

vocabulary


# ### Cumulative frequency
# 
# Do any words produced in the last example help us grasp the topic or genre of this text? (Eliminate English "plumbing")

# In[ ]:

fdist.plot(5, cumulative = True)

