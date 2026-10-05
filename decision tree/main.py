import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import re

SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account",
                    "bank", "confirm", "signin", "password", "paypal"]


def urlConvertor(url):
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower()
    return {
        'url_length' : len(url),
        "host_length": len(host),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_slashes": url.count("/"),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", host))),
        "uses_https": int(url.lower().startswith("https")),
        "num_subdomains": max(host.count(".") - 1, 0),
        "num_suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "has_port": int(":" in host),
    }

def giniImpurity(leaves):
    return (1-np.sum(leaves ** 2)) 

dataframe = pd.read_csv("phishing_ste_urls.csv")
dataframe["Label"] = (dataframe["Label"].str.lower() == "bad" ).astype(int)

X = pd.DataFrame([urlConvertor(u) for u in dataframe["URL"] ])
y = dataframe["Label"]

X = X.to_numpy()

X_train  , X_test ,  y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

best_gini = float("inf")
best_split = {}

for feature_idx in range(X.shape[1]):
    thresholds = np.unique(X[:, feature_idx])

    for threshold in thresholds:

        left_mask = X[:, feature_idx] <= threshold

        y_left = y_train[left_mask]
        y_right = y_train[~left_mask]

        p_class_1 = np.mean(y_left == 1)
        p_class_0 = 1 - p_class_1

        leaves = np.array([p_class_0, p_class_1])
        impurity_left = giniImpurity(leaves)

        p_class_2 = np.mean(y_right == 1)
        p_class_3 = 1 - p_class_2

        leaves = np.array([p_class_2, p_class_3])
        impurity_right = giniImpurity(leaves)

        n_total = len(y_train)
        n_left = len(y_left)
        n_right = len(y_right)

        weighted_gini = (n_left / n_total) * impurity_left + (
            n_right / n_total
        ) * impurity_right

        if weighted_gini < best_gini:

            best_gini = weighted_gini

            best_split = {
                "feature_idx": feature_idx,
                "threshold": threshold,
                "weighted_gini": weighted_gini,
            }

print(best_split)