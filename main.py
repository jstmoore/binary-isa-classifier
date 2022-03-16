import requests
import logging
import base64
import binascii
import time
import pickle
import joblib
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        failcnt = 0
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                failcnt = 0

                return r.json()
            except Exception as e:
                failcnt += 1
                if failcnt == 5:
                    break
                self.log.error(e)
                self.log.info('Waiting 10 seconds before next request')
                time.sleep(5)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r

if __name__ == "__main__":

    vectorizer = joblib.load('./models/vectorizer')
    model_names = ['linearSVC', 'rbfSVC', 'perceptron_sgd', 'mhuber_sgd']
    models = [joblib.load(f'./models/{name}') for name in model_names]

    def mode(arr: list):
        vals, cnts = np.unique(arr, return_counts=True)
        best = (vals[0], cnts[0])
        for val, cnt in zip(vals[1:], cnts[1:]):
            if cnt > best[1]:
                best = (val, cnt)
        return best[0]

    def predict(binary):
        x = str(binary)
        x = x[x.find('b\'') + 2:].encode()
        v = vectorizer.transform([x])
        predictions = [model.predict(v)[0] for model in models]
        # if len(np.unique(predictions)[0]) > 1:
        #     print(predictions)
        #     print("disagreement")
        return mode(predictions)

    # create the server object
    s = Server()
    N = 500000
    df_path = "binarydata2.csv"
    df = pd.read_csv(df_path, index_col=0)
    data = []

    for i in range(N):
        # query the /challenge endpoint
        s.get()

        # choose a random target and /solve
        target = predict(binascii.hexlify(s.binary))
        s.post(target)

        if s.ans != target:
            data.append([s.ans, s.binary])

        # s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(target, s.ans, s.wins))

        # 500 consecutive correct answers are required to win
        # very very unlikely with current code
        if s.hash:
            s.log.info("You win! {}".format(s.hash))
            print("\n"*2 + "HASH::\t" + s.hash)
            print("Submit the above hash to Praetorian")

        if i > 0 and not i % 1000:
            new_df = pd.DataFrame(data, columns=['label', 'data'])
            df = pd.concat([df, new_df], axis=0)
            df.reset_index(inplace=True, drop=True)
            df.to_csv("binarydata2.csv")
            print(f"{N - i - 1} iterations remaining...")
            print()
    
    df.to_csv("binarydata2.csv")
