import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# setting dimensions of the data table
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 750)

# loading dataset into Python
path_to_file = "C:/Users/Yeeva/IdeaProjects/F1_Data_Analysis/F1Drivers_Dataset_upto_2022.csv"
df = pd.read_csv(path_to_file, encoding="ISO-8859-1")

# grabs data from all the rows only from the columns specified
df = df.loc[:, ["Race_Starts", "Pole_Rate", "Win_Rate", "Podium_Rate", "Points_Per_Entry", "Years_Active", "Champion"]]

# sets the column data for X and y
# row number, col number
X = df.iloc[:, 0:6]
y = df.iloc[:, 6]
#print(X, "\n", y)

# NOTE: Whereas df.loc grabs data based on column name, df.iloc grabs data based on column index

# splits the data into training and testing data. Allocates 20% of the data for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

# selects the standardization method for normalizing data and sets data to be in range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train)  # 80%
X_test_scaled = scaler.fit_transform(X_test)  # 20%

# looks at 8 closest neighbors
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Score:", knn.score(X_test_scaled, y_test))

rfc = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_split=5, min_samples_leaf=5, criterion="entropy")
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print("RFC Score:", rfc.score(X_test, y_test))

log = LogisticRegression()
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)
print("Logistic Regression Score:", log.score(X_test, y_test))


cm_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN CM:", cm_knn)

cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print("RFC CM:", cm_rfc)

""" CONFUSION MATRIX:

    [[True Neg      False Pos
     [False Neg     True Pos]]
     
"""

cr_knn = classification_report(y_test, y_pred_knn)
print(cr_knn)
# FINDING: KNN has tendency to give false positives

cr_rfc = classification_report(y_test, y_pred_rfc)
print(cr_rfc)
# FINDING: RFC has tendency to give false negatives

def knn_predict(race_starts, pole_num, win_num, podium_num, points_num, years_active):
    pole_rate = pole_num / race_starts
    win_rate = win_num / race_starts
    podium_rate = podium_num / race_starts
    points_per_entry = points_num / race_starts
    data = [[race_starts, pole_rate, win_rate, podium_rate, points_per_entry, years_active]]
    normalized_data = scaler.transform(pd.DataFrame(data, columns=["Race_Starts", "Pole_Rate", "Win_Rate",
                                                                   "Podium_Rate", "Points_Per_Entry",
                                                                   "Years_Active"]))
    print(knn.predict(normalized_data))


def rfc_predict(race_starts, pole_num, win_num, podium_num, points_num, years_active):
    pole_rate = pole_num / race_starts
    win_rate = win_num / race_starts
    podium_rate = podium_num / race_starts
    points_per_entry = points_num / race_starts
    data = [[race_starts, pole_rate, win_rate, podium_rate, points_per_entry, years_active]]
    pd_data = pd.DataFrame(data, columns=["Race_Starts", "Pole_Rate", "Win_Rate", "Podium_Rate", "Points_Per_Entry",
                                                                   "Years_Active"])
    print(rfc.predict(pd_data))

#knn_predict(114, 5, 5, 17, 159.5, 9)
#rfc_predict(114, 5, 5, 17, 159.5, 9)
