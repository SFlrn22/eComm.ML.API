import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

with open('./new-books', 'rb') as file:
    books = pickle.load(file)
with open('./new-ratings', 'rb') as file:
    ratings = pickle.load(file)
with open('./new-users', 'rb') as file:
    users = pickle.load(file)

user_ratings = users.merge(ratings,on="User-ID")
merged_dataset = books.merge(user_ratings, on="ISBN")
books_with_ratings = ratings.merge(books, on="ISBN")

ratings = ratings.rename(columns={'Book-Rating':'Rating'})
books = books.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author'})
merged_dataset = merged_dataset.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'User-ID': 'UserId', 'Book-Rating':'Rating'})
books_with_ratings = books_with_ratings.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'User-ID': 'UserId', 'Book-Rating':'Rating'})

def BookSimilarityInit(books_with_ratings):
    ratings_count = books_with_ratings.groupby('UserId').count()['ISBN']
    ratings_count = ratings_count.sort_values(ascending=False).reset_index()
    ratings_count.rename(columns={'ISBN':'NumberOfBooks'}, inplace=True)

    filtered_users = ratings_count[ratings_count['NumberOfBooks'] >= 1]
    books_ratings_count = ratings.merge(books, on="ISBN").groupby('Title').count()['ISBN'].sort_values(ascending=False).reset_index()
    books_ratings_count.rename(columns={'ISBN':'NumberOfRatings'}, inplace=True)

    filtered_books = books_ratings_count[books_ratings_count['NumberOfRatings']>=1]
    books_with_filtered_users = pd.merge(filtered_users, books_with_ratings, on='UserId')

    filtered_data = pd.merge(filtered_books, books_with_filtered_users, on='Title')

    pivot_table = filtered_data.pivot_table(index='ISBN',columns='UserId', values='Rating')
    pivot_table.fillna(0, inplace=True)

    similarity_array = cosine_similarity(pivot_table)

    return pivot_table, similarity_array

def FilteringCollaborationRecommendations(isbn, dataset):
    isbnList = []
    pivot_table, similarity_array = BookSimilarityInit(dataset)
    # np.where(pivot_table.index==isbn)[0][0]
    for book in sorted(list(enumerate(similarity_array[np.where(pivot_table.index==isbn)[0][0]])), key=lambda x: x[1], reverse=True)[1:11]:
        isbnList.append(pivot_table.index[book[0]])

    return isbnList

def ContentBasedRecommendations(isbn, merged_dataset):
    book = merged_dataset.loc[merged_dataset['ISBN'] == isbn]['ISBN']

    book_isbn = "";

    count_rate = pd.DataFrame(merged_dataset['ISBN'].value_counts())
    rare_books = count_rate[count_rate["count"]<5].index
    common_books = merged_dataset[~merged_dataset["ISBN"].isin(rare_books)]

    r_books=[]

    common_books=common_books.drop_duplicates(subset=["ISBN"])

    if(book.empty):
        book = books.loc[books['ISBN'] == isbn]
        book_isbn = book.iloc[0][0]
        common_books = pd.concat([common_books,book], ignore_index = False)
    else:
        book_isbn = book.iloc[0]

    if book_isbn in rare_books:
        book = books.loc[books['ISBN'] == isbn]
        book_isbn = book.iloc[0][0]
        common_books = pd.concat([common_books,book], ignore_index = False)

    common_books.reset_index(inplace=True)
    
    common_books["index"]=[i for i in range(common_books.shape[0])]
    common_books['ISBN'] = common_books['ISBN'].astype('object')
    common_books['Title'] = common_books['Title'].astype('object')
    common_books['Author'] = common_books['Author'].astype('object')
    common_books['Publisher'] = common_books['Publisher'].astype('object')


    targets=["Title","Author","Publisher"]
    common_books["all_features"] = [" ".join(common_books[targets].astype(str).iloc[i,].values) for i in range(common_books[targets].shape[0])]

    common_books=common_books.drop_duplicates(subset=["ISBN"])
    common_books.reset_index(inplace=True)
    
    vectorizer=CountVectorizer()
    common_booksVector=vectorizer.fit_transform(common_books["all_features"])

    similarity=cosine_similarity(common_booksVector)
    index=common_books[common_books["ISBN"]==book_isbn]["index"].values[0]
    similar_books=list(enumerate(similarity[index]))
    similar_booksSorted=sorted(similar_books,key=lambda x:x[1],reverse=True)[1:11]
    
    for i in range(len(similar_booksSorted)):
        r_books.append(common_books[common_books["index"]==similar_booksSorted[i][0]]["ISBN"].item())

    return r_books

def get_recommendations_batch(isbn_batch):
    return [(isbn, ContentBasedRecommendations(isbn, merged_dataset)) for isbn in isbn_batch]

def get_recommendations_user_batch(isbn_batch):
    return [(isbn, FilteringCollaborationRecommendations(isbn, books_with_ratings)) for isbn in isbn_batch]

if __name__ == '__main__':
    random_isbns = books['ISBN'].tolist()
    halfBooksRecommendationList = []

    # Create batches of 1000 ISBNs
    batch_size = 2000
    isbn_batches = [random_isbns[i:i + batch_size] for i in range(0, len(random_isbns), batch_size)]

    # Use ProcessPoolExecutor with batches
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(get_recommendations_user_batch, batch): idx for idx, batch in enumerate(isbn_batches)}

        for future in as_completed(futures):
            try:
                recommendations_batch = future.result()
                halfBooksRecommendationList.extend(recommendations_batch)  # Flatten the list
            except Exception as e:
                print(f"Error processing batch {futures[future]}: {e}")

            # Progress tracking
            if (futures[future] + 1) % 2 == 0:  # Every 2 batches
                print(f"Reached batch index {(futures[future] + 1) * batch_size}")

    # Save the final recommendations list
    pickle.dump(halfBooksRecommendationList, open('./new_recommendations_filtering', 'wb'))