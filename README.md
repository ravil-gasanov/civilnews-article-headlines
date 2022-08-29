# Article Headline modelling
I have scraped article data from civil.ge to:
(1) predict views from headlines (e.g. using Word2Vec embeddings)
(2) model headline topics (e.g. using Bayesian SMM) and use the inferred topics and the number of views to characterize articles that seem to gain most reads/clicks.

However, preliminary results were too poor to proceed. Namely, the headline embeddings do not seem to be predictive of the number of views. Perhaps more advanced embedding models or building a custom neural network predictor could yield better results, but for now I have decided to drop the project.
