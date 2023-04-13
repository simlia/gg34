import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
#df.info()
#print(df)
def dia():
    muzh = df[df['sex']== 2 ]['result'].mean() * 100
    zhen = df[df['sex']== 1]['result'].mean() * 100
    s = pd.Series(data = [muzh, zhen], index = ['мужчины', "женщины"])
    s.plot(kind = 'barh', figsize = (8, 5), grid = True,)
    plt.show()

    unik = df[df['occupation_type']== 'university' ]['result'].mean() * 100
    work = df[df['occupation_type']== 'work' ]['result'].mean() * 100
    s = pd.Series(data = [unik, work], index = ['университет', "работа"])
    s.plot(kind = 'barh', figsize = (8, 5), grid = True,)
    plt.show()

temp = df.pivot_table(columns = 'occupation_type',
index = 'sex', values = 'result', aggfunc = 'mean')
print(temp)

df.drop(['id', 'bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation',
'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start',
'career_end'], axis=1, inplace=True)


def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

#print(df['education_status'].value_counts())
def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == "Student (Specialist)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1
    elif edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    else:
        return 3

df['education_status'] = df['education_status'].apply(edu_status_apply)

def langs_apply(langs):
    if langs.find('Русский') != -1 and langs.find('English') != -1:
        return 1
    else:
        return 0 

df['langs'] = df['langs'].apply(langs_apply)
df['occupation_type'].fillna('university', inplace = True)

def occupation_type_apply(occupation_type):
    if occupation_type == 'university':
        return 1
    elif occupation_type == 'work':
        return 0

df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)

df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))
