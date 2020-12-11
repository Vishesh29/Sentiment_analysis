Essentially, sentiment analysis or sentiment classification fall into the broad category of text classification tasks where we
are supplied with a phrase, or a list of phrases and our classifier is supposed to tell if the sentiment behind that is
positive, negative or neutral. Sometimes, the third attribute is not taken to keep it a binary classification problem. 
In recent tasks, sentiments like "somewhat positive" and "somewhat negative" are also being considered.
The phrases correspond to short movie reviews, and each one of them conveys different sentiments.


About the Dataset-

The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.

-train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
-test.tsv contains just phrases. You must assign a sentiment label to each phrase.

In this project The sentiment labels are:
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive



