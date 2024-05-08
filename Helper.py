import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from spello.model import SpellCorrectionModel
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import warnings
import pickle
from scipy.spatial import distance
from sklearn.decomposition import PCA
from PIL import Image
import whisper
import torch
import io
from diffusers import StableDiffusionPipeline, UniDiffuserPipeline
import matplotlib.pyplot as plt
import base64
from subprocess import CalledProcessError, run

from dotenv import dotenv_values
from hashlib import sha256

secrets = dotenv_values(".env")

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model('large').to(device)

sp = SpellCorrectionModel(language='en')
sp.load("./Utils/model.pkl")

pickled_features = pickle.load(open('./Utils/pca_features_rated_plus_save', 'rb'))
pickled_pca = pickle.load(open('./Utils/pca_rated_plus_save', 'rb'))
pickled_extractor = pickle.load(open('./Utils/extractor_model_rated_plus_save', 'rb'))

path = "./Images"
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames]
images = [image.replace("\\", "/") for image in images]

books = pd.read_csv("./Datasets/products.csv")
ratings = pd.read_csv("./Datasets/ratings.csv")
users = pd.read_csv("./Datasets/users.csv")

stableDiffusionModel = "dreamlike-art/dreamlike-diffusion-1.0"
stableDiffusionPipe = StableDiffusionPipeline.from_pretrained(stableDiffusionModel, torch_dtype=torch.float16, use_safetensors=True)
stableDiffusionPipe = stableDiffusionPipe.to(device)

# itt_model = "thu-ml/unidiffuser-v1"
# itt_pipe = UniDiffuserPipeline.from_pretrained(itt_model, torch_dtype=torch.float16)
# itt_pipe.to(device)

def Authenticate(credentials):
    user = base64.b64encode(sha256(secrets["API_USERNAME"].encode('utf-8')).digest())
    password = base64.b64encode(sha256(secrets["API_PASSWORD"].encode('utf-8')).digest())
    
    if(user.decode('utf-8') == credentials.username and password.decode('utf-8') == credentials.password):
        return True

    return False

def TextToImage(title):
    prompt = f"{title} in dreamlike art"
    image = stableDiffusionPipe(prompt).images[0];
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def InitializeDataset():
    user_ratings = users.merge(ratings,on="UserId")
    merged_dataset = books.merge(user_ratings, on="ISBN")
    books_with_ratings = ratings.merge(books, on="ISBN")
    return merged_dataset, books_with_ratings

def BookSimilarityInit(books_with_ratings):
    ratings_count = books_with_ratings.groupby('UserId').count()['ISBN']
    ratings_count = ratings_count.sort_values(ascending=False).reset_index()
    ratings_count.rename(columns={'ISBN':'NumberOfBooks'}, inplace=True)

    filtered_users = ratings_count[ratings_count['NumberOfBooks'] >= 50]
    books_ratings_count = ratings.merge(books, on="ISBN").groupby('Title').count()['ISBN'].sort_values(ascending=False).reset_index()
    books_ratings_count.rename(columns={'ISBN':'NumberOfRatings'}, inplace=True)

    filtered_books = books_ratings_count[books_ratings_count['NumberOfRatings']>=20]
    books_with_filtered_users = pd.merge(filtered_users, books_with_ratings, on='UserId')

    filtered_data = pd.merge(filtered_books, books_with_filtered_users, on='Title')

    pivot_table = filtered_data.pivot_table(index='ISBN',columns='UserId', values='Rating')
    pivot_table.fillna(0, inplace=True)

    similarity_array = cosine_similarity(pivot_table)

    return pivot_table, similarity_array

def UserSimilarityInit(books_with_ratings):
    ratings_count = books_with_ratings.groupby('UserId').count()['ISBN']
    ratings_count = ratings_count.sort_values(ascending=False).reset_index()
    ratings_count.rename(columns={'ISBN':'NumberOfBooks'}, inplace=True)

    filtered_users = ratings_count[ratings_count['NumberOfBooks'] >= 50]
    books_ratings_count = ratings.merge(books, on="ISBN").groupby('Title').count()['ISBN'].sort_values(ascending=False).reset_index()
    books_ratings_count.rename(columns={'ISBN':'NumberOfRatings'}, inplace=True)

    filtered_books = books_ratings_count[books_ratings_count['NumberOfRatings']>=20]
    books_with_filtered_users = pd.merge(filtered_users, books_with_ratings, on='UserId')

    filtered_data = pd.merge(filtered_books, books_with_filtered_users, on='Title')

    pivot_table = filtered_data.pivot_table(index='UserId',columns='Title', values='Rating')
    pivot_table.fillna(0, inplace=True)

    correlations = pivot_table.corr(method='pearson')

    return correlations

def FilteringCollaborationRecommendations(isbn, dataset):
    isbnList = []
    pivot_table, similarity_array = BookSimilarityInit(dataset)
    # np.where(pivot_table.index==isbn)[0][0]
    for book in sorted(list(enumerate(similarity_array[np.where(pivot_table.index==isbn)[0][0]])), key=lambda x: x[1], reverse=True)[1:11]:
        isbnList.append(pivot_table.index[book[0]])

    return isbnList

def UserSimilarityRecommendations(userId):
    correlations = UserSimilarityInit(books_with_ratings)
    # preia carti cu rating mare citite de user
    # similar = correlations[''].sort_values(ascending=False)

def GetTopTenRecommendations(merged_dataset):
    books_top_rated = merged_dataset['ISBN'].value_counts().head(30)
    books_top_rated = list(books_top_rated.index)

    highestRatingBooks = pd.DataFrame(columns = merged_dataset.columns)

    for book in books_top_rated:

        cond_df = merged_dataset[merged_dataset['ISBN'] == book]
        highestRatingBooks =  pd.concat([highestRatingBooks, cond_df], axis=0)


    highestRatingBooks = highestRatingBooks[highestRatingBooks['Rating'] != 0]
    highestRatingBooks = highestRatingBooks.groupby('ISBN')['Rating'].agg('mean').reset_index().sort_values(by='Rating', ascending=False)

    isbnList = []

    for isbn in highestRatingBooks.head(10)['ISBN']:
        isbnList.append(isbn)

    return isbnList

def ItemBasedRecommendations(isbn, merged_dataset):
    book = merged_dataset.loc[merged_dataset['ISBN'] == isbn]['ISBN']

    book_title = book.iloc[0]

    count_rate = pd.DataFrame(merged_dataset['ISBN'].value_counts())
    rare_books = count_rate[count_rate["count"]<=100].index

    common_books = merged_dataset[~merged_dataset["ISBN"].isin(rare_books)]

    similar_books = {}

    recommended_list = []

    if book_title in rare_books:
        recommended_list.append("this is a rare book")
        return recommended_list
    else:

        item_based_cb = common_books.pivot_table(index=["UserId"],columns=["ISBN"],values="Rating")
        sim = item_based_cb[book_title]
        recommendation_df=pd.DataFrame(item_based_cb.corrwith(sim).sort_values(ascending=False)).reset_index(drop=False)

        if not recommendation_df['ISBN'][recommendation_df['ISBN'] == book_title].empty:
            recommendation_df=recommendation_df.drop(recommendation_df[recommendation_df["ISBN"]==book_title].index[0])

        less_rating=[]
        for i in recommendation_df["ISBN"]:
            if merged_dataset[merged_dataset["ISBN"]==i]["Rating"].mean() < 5:
                less_rating.append(i)

        if recommendation_df.shape[0] - len(less_rating) > 5:

            recommendation_df = recommendation_df[~recommendation_df["ISBN"].isin(less_rating)]
            recommendation_df.columns=["ISBN","Correlation"]

        for (candidate_book, corr) in zip(recommendation_df['ISBN'], recommendation_df['Correlation']):
            corr_thershold = 0.7
            if corr > corr_thershold:
                similar_books[candidate_book] = corr
            else:
                break
        
        sorted_books = sorted(similar_books.items(), key=lambda x: x[1], reverse=True)

        for book in sorted_books[1:10]:
            recommended_list.append(book[0])

        return recommended_list

def ContentBasedRecommendations(isbn, merged_dataset):
    book = merged_dataset.loc[merged_dataset['ISBN'] == isbn]['ISBN']

    book_isbn = "";

    count_rate = pd.DataFrame(merged_dataset['ISBN'].value_counts())
    rare_books = count_rate[count_rate["count"]<20].index
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
    common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]

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

def LoadAudio(bytearr, sr):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, input=bytearr, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data

def GetTextFromVoice(content):
    audio_data = np.frombuffer(content, np.int16).flatten().astype(np.float32) / 32768.0
    audio = LoadAudio(content, 16000)
    result = model.transcribe(audio)
    #spellchecked = SpellCheck(result['text'].strip())
    return result['text'].strip()

def SpellCheck(title):
    return sp.spell_correct(title)['spell_corrected_text']

def SearchByImage(input):
    img, x = LoadAndProcessImage(input);
    feature = pickled_extractor.predict(x)
    pca_feature = pickled_pca.transform(feature)[0]
    similar_idx = [ distance.cosine(pca_feature, feat) for feat in pickled_features ]
    idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[0:10]
    return images[idx_closest[0]].replace("./Images/", "")

def LoadAndProcessImage(byteArr):
    img = Image.open(io.BytesIO(byteArr))
    img = img.convert('RGB')
    img = img.resize(pickled_extractor.input_shape[1:3])
    # img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def LoadPIL(byteArr):
    img = Image.open(io.BytesIO(byteArr))
    img = img.convert('RGB')
    img = img.resize((512, 512))

    return img;

# def ImageToText(input):
#     #init_image = load_image(input).resize((512, 512))
#     init_image = LoadPIL(input)
#     sample = itt_pipe(image=init_image, num_inference_steps=5)
#     i2t_text = sample.text[0]
#     return i2t_text