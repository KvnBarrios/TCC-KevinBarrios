import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Carregue os dados do arquivo CSV em um DataFrame do pandas
df = pd.read_csv('data.csv')

# Converta o DataFrame em uma lista de listas de tuplas
sentences = df.groupby('Sentença')[['Palavra', 'Rótulo']].apply(lambda x: [tuple(x) for x in x.values]).tolist()

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for word, label, postag in sent]

# Separe os dados em treino e teste
sentences_train, sentences_test = train_test_split(sentences, test_size=0.2)

# Preparação dos dados para o modelo
X_train = [sent2features(s) for s in sentences_train]
y_train = [sent2labels(s) for s in sentences_train]

X_test = [sent2features(s) for s in sentences_test]
y_test = [sent2labels(s) for s in sentences_test]

# Criação do modelo CRF
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# Treinamento do modelo
crf.fit(X_train, y_train)

# Avaliação do modelo
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred))
