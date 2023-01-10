# first section, explain how packages work
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# save for later, add after fifth section
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load

# second section, explain loading the data and running a visualization
pokemon = pd.read_csv("pokemon.csv")
sns.barplot(x="Type 1", y="HP", data=pokemon)
# plt.show()

# third section, just say pull in combats now
battles = pd.read_csv("combats.csv")

poke_battles = battles.merge(pokemon, left_on="First_pokemon", right_on="#")
poke_battles = poke_battles.merge(pokemon, left_on="Second_pokemon", right_on="#")
poke_battles["Target"] = poke_battles["Winner"] == poke_battles["First_pokemon"]

# fourth section, preprocessing
poke_battles = poke_battles.drop(["#_x", "Name_x", "#_y", "Name_y"], axis=1)
encoder = LabelEncoder()
poke_battles["Type_1_x"] = encoder.fit_transform(poke_battles["Type 1_x"])
poke_battles["Type_2_x"] = encoder.fit_transform(poke_battles["Type 2_x"])
poke_battles["Type_1_y"] = encoder.fit_transform(poke_battles["Type 1_y"])
poke_battles["Type_2_y"] = encoder.fit_transform(poke_battles["Type 2_y"])


poke_battles = pd.get_dummies(
    poke_battles, columns=["Type 1_x", "Type 2_x", "Type 1_y", "Type 2_y"]
)

# fifth section, train test split
X = poke_battles.drop(["Winner", "Target"], axis=1)
y = poke_battles.Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# sixth section, train the model and evaluate
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(accuracy_score(y_test, y_pred))

# seventh section, save the model
dump(rf, open("poke_model.pkl", "wb"))

# eighth section, load the model and make a prediction
model = load(open("poke_model.pkl", "rb"))
print(model.predict(X_test.iloc[0:1]))
print(X_test.iloc[0:1])


# # ninth section, make a prediction
# def predict_winner(pokemon_1, pokemon_2):
#     pokemon_1 = pokemon[pokemon['Name'] == pokemon_1]
#     pokemon_2 = pokemon[pokemon['Name'] == pokemon_2]
#     pokemon_1 = pokemon_1.drop(['#', 'Name', 'Type 1', 'Type 2'], axis=1)
#     pokemon_2 = pokemon_2.drop(['#', 'Name', 'Type 1', 'Type 2'], axis=1)
#     pokemon_1 = pd.get_dummies(pokemon_1, columns=['Type 1', 'Type 2'])
#     pokemon_2 = pd.get_dummies(pokemon_2, columns=['Type 1', 'Type 2'])
#     pokemon_1 = pokemon_1.reindex(columns=pokemon_2.columns, fill_value=0)
#     pokemon_2 = pokemon_2.reindex(columns=pokemon_1.columns, fill_value=0)
#     pokemon_1 = pokemon_1.append(pokemon_2)
#     pokemon_1 = pokemon_1.reset_index(drop=True)
#     prediction = model.predict(pokemon_1)
#     if prediction[0] == 1:
#         return pokemon_1['Name'][0]
#     else:
#         return pokemon_1['Name'][1]
