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
poke_battles = poke_battles.drop(["First_pokemon", "Second_pokemon", "Winner", "#_x", "Name_x", "#_y", "Name_y"], axis=1)
encoder = LabelEncoder()
poke_battles["Type_1_x_Encode"] = encoder.fit_transform(poke_battles["Type 1_x"])
poke_battles["Type_2_x_Encode"] = encoder.fit_transform(poke_battles["Type 2_x"])
poke_battles["Type_1_y_Encode"] = encoder.fit_transform(poke_battles["Type 1_y"])
poke_battles["Type_2_y_Encode"] = encoder.fit_transform(poke_battles["Type 2_y"])
poke_battles = poke_battles.drop(["Type 1_x", "Type 2_x", "Type 1_y", "Type 2_y"], axis=1)

# fifth section, train test split
X = poke_battles.drop(["Target"], axis=1)
y = poke_battles.Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# sixth section, train the model and evaluate
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# seventh section, save the model
dump(rf, open("poke_model.pkl", "wb"))

# eighth section, load the model and make a prediction
model = load(open("poke_model.pkl", "rb"))


# ninth section, make a prediction
def predict_winner(pokemon_1, pokemon_2):
    poke_df_x = pokemon[pokemon["Name"] == pokemon_1]
    poke_df_y = pokemon[pokemon["Name"] == pokemon_2]
    poke_df_x.columns = [col + "_x" for col in poke_df_x.columns]
    poke_df_y.columns = [col + "_y" for col in poke_df_y.columns]
    poke_df = pd.concat([poke_df_x, poke_df_y], axis=1)
    poke_df = poke_df.drop(["#_x", "Name_x", "#_y", "Name_y"], axis=1)
    poke_df["Type_1_x_Encode"] = encoder.transform(poke_df["Type 1_x"])
    poke_df["Type_2_x_Encode"] = encoder.transform(poke_df["Type 2_x"])
    poke_df["Type_1_y_Encode"] = encoder.transform(poke_df["Type 1_y"])
    poke_df["Type_2_y_Encode"] = encoder.transform(poke_df["Type 2_y"])
    poke_df = poke_df.drop(["Type 1_x", "Type 2_x", "Type 1_y", "Type 2_y"], axis=1)
    return model.predict(poke_df)[0]

print(predict_winner("Charizard", "Blastoise"))
