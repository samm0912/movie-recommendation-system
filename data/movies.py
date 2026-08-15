"""
Sample movie dataset — mimics MovieLens structure.
In a real project, replace with actual MovieLens CSV files.
"""

MOVIES = [
    {"id": 1,  "title": "The Dark Knight",       "genres": "Action|Crime|Drama",        "year": 2008, "rating": 9.0, "poster": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
    {"id": 2,  "title": "Inception",              "genres": "Action|Sci-Fi|Thriller",    "year": 2010, "rating": 8.8, "poster": "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg"},
    {"id": 3,  "title": "Interstellar",           "genres": "Adventure|Drama|Sci-Fi",    "year": 2014, "rating": 8.6, "poster": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"},
    {"id": 4,  "title": "The Shawshank Redemption","genres": "Drama|Crime",              "year": 1994, "rating": 9.3, "poster": "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"},
    {"id": 5,  "title": "Pulp Fiction",           "genres": "Crime|Drama|Thriller",      "year": 1994, "rating": 8.9, "poster": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"},
    {"id": 6,  "title": "The Matrix",             "genres": "Action|Sci-Fi",             "year": 1999, "rating": 8.7, "poster": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    {"id": 7,  "title": "Forrest Gump",           "genres": "Drama|Romance",             "year": 1994, "rating": 8.8, "poster": "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg"},
    {"id": 8,  "title": "Avengers: Endgame",      "genres": "Action|Adventure|Sci-Fi",   "year": 2019, "rating": 8.4, "poster": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"},
    {"id": 9,  "title": "Joker",                  "genres": "Crime|Drama|Thriller",      "year": 2019, "rating": 8.5, "poster": "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg"},
    {"id": 10, "title": "Parasite",               "genres": "Drama|Thriller",            "year": 2019, "rating": 8.6, "poster": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"},
    {"id": 11, "title": "Dune",                   "genres": "Adventure|Drama|Sci-Fi",    "year": 2021, "rating": 8.0, "poster": "https://image.tmdb.org/t/p/w500/d5NXSklpcuveafzltaU2fPDaIXX.jpg"},
    {"id": 12, "title": "Spider-Man: No Way Home","genres": "Action|Adventure|Sci-Fi",   "year": 2021, "rating": 8.3, "poster": "https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg"},
    {"id": 13, "title": "The Godfather",          "genres": "Crime|Drama",               "year": 1972, "rating": 9.2, "poster": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsLe1rhdNatFX.jpg"},
    {"id": 14, "title": "Fight Club",             "genres": "Drama|Thriller",            "year": 1999, "rating": 8.8, "poster": "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"},
    {"id": 15, "title": "Goodfellas",             "genres": "Biography|Crime|Drama",     "year": 1990, "rating": 8.7, "poster": "https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg"},
    {"id": 16, "title": "The Silence of the Lambs","genres":"Crime|Drama|Thriller",      "year": 1991, "rating": 8.6, "poster": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg"},
    {"id": 17, "title": "Gladiator",              "genres": "Action|Adventure|Drama",    "year": 2000, "rating": 8.5, "poster": "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"},
    {"id": 18, "title": "The Lion King",          "genres": "Animation|Adventure|Drama", "year": 1994, "rating": 8.5, "poster": "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg"},
    {"id": 19, "title": "Titanic",                "genres": "Drama|Romance",             "year": 1997, "rating": 7.9, "poster": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"},
    {"id": 20, "title": "Avatar",                 "genres": "Action|Adventure|Sci-Fi",   "year": 2009, "rating": 7.8, "poster": "https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg"},
    {"id": 21, "title": "Whiplash",               "genres": "Drama|Music",               "year": 2014, "rating": 8.5, "poster": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg"},
    {"id": 22, "title": "La La Land",             "genres": "Drama|Music|Romance",       "year": 2016, "rating": 8.0, "poster": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg"},
    {"id": 23, "title": "Get Out",                "genres": "Horror|Mystery|Thriller",   "year": 2017, "rating": 7.7, "poster": "https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg"},
    {"id": 24, "title": "Us",                     "genres": "Horror|Mystery|Thriller",   "year": 2019, "rating": 6.8, "poster": "https://image.tmdb.org/t/p/w500/ux2maFlzqiy9VMqkTQBQ6vfBwsY.jpg"},
    {"id": 25, "title": "1917",                   "genres": "Drama|War",                 "year": 2019, "rating": 8.3, "poster": "https://image.tmdb.org/t/p/w500/iZf0KyrE25z1sage4SYFLCCrMi9.jpg"},
]

# Simulated user ratings (user_id, movie_id, rating)
RATINGS = [
    (1,1,5),(1,2,5),(1,3,4),(1,6,5),(1,8,4),(1,11,5),(1,12,4),
    (2,4,5),(2,5,5),(2,13,5),(2,15,5),(2,16,4),(2,9,4),(2,10,5),
    (3,7,5),(3,18,5),(3,19,5),(3,22,4),(3,21,5),(3,14,3),
    (4,1,4),(4,2,4),(4,6,5),(4,17,5),(4,20,4),(4,25,5),(4,8,5),
    (5,4,5),(5,13,5),(5,5,4),(5,14,4),(5,9,5),(5,16,5),(5,15,5),
    (6,3,5),(6,11,5),(6,2,4),(6,6,4),(6,20,5),(6,12,4),(6,1,3),
    (7,23,5),(7,24,4),(7,16,5),(7,9,4),(7,5,4),(7,10,5),
    (8,7,5),(8,19,5),(8,22,5),(8,21,4),(8,18,4),(8,4,3),
    (9,25,5),(9,17,4),(9,1,4),(9,8,5),(9,12,5),(9,3,4),
    (10,10,5),(10,9,5),(10,23,4),(10,5,4),(10,13,4),(10,14,5),
]
